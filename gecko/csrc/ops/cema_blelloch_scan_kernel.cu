#include <ATen/AccumulateType.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/complex.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <type_traits>

#include "blas.h"
#include "complex_utils.cuh"
#include "cuda_utils.cuh"
#include "ops/cema_blelloch_scan.h"
#include "reduce.cuh"
#include "utils.h"

namespace gecko {
namespace ops {
namespace {
constexpr int64_t kMaxNumThreads = 256;

struct ComplexRightAffineProd {
    __device__ __forceinline__ float4 identity() const {
        return make_float4(1.f, 0.f, 0.f, 0.f);
    }
    __device__ __forceinline__ float4 operator()(const float4 &first, const float4 &second) const {
        float first_real = first.x * second.x - first.y * second.y;
        float first_imag = first.x * second.y + first.y * second.x;
        float second_real = second.x * first.z - second.y * first.w + second.z;
        float second_imag = second.x * first.w + second.y * first.z + second.w;
        return make_float4(first_real, first_imag, second_real, second_imag);
    }
};

struct ComplexLeftAffineProd {
    __device__ __forceinline__ float4 identity() const {
        return make_float4(1.f, 0.f, 0.f, 0.f);
    }
    __device__ __forceinline__ float4 operator()(const float4 &first, const float4 &second) const {
        float first_real = first.x * second.x - first.y * second.y;
        float first_imag = first.x * second.y + first.y * second.x;
        float second_real = first.x * second.z - first.y * second.w + first.z;
        float second_imag = first.x * second.w + first.y * second.z + first.w;
        return make_float4(first_real, first_imag, second_real, second_imag);
    }
};


template <typename T, typename T_ACC, int64_t CHUNK=256>
__global__ void CEMABlellochScanCUDAFwdKernel(int64_t D, int64_t N, int64_t L,
                                              const T* __restrict__ x,
                                              const T_ACC* __restrict__ p,
                                              const c10::complex<T_ACC>* __restrict__ q,
                                              const c10::complex<T_ACC>* __restrict__ gamma,
                                              const bool* __restrict__ bos_mask,
                                              const c10::complex<T_ACC>* __restrict__ h0,
                                              c10::complex<T_ACC>* __restrict__ h,
                                              T_ACC* __restrict__ y,
                                              c10::complex<T_ACC>* __restrict__ chunk_decay,
                                              c10::complex<T_ACC>* __restrict__ chunk_gain) {
    const int64_t b = blockIdx.x;
    const int64_t d = blockIdx.y;
    const int64_t n = blockIdx.z;
    const int64_t tid = threadIdx.x;

    if (d >= D || n >= N) return;

    const int64_t num_chunks = (L + CHUNK - 1) / CHUNK;

    extern __shared__ float4 fwd_scan_shm[];
    float4* shared_result = fwd_scan_shm;

    // Init & load
    ComplexRightAffineProd prefixOp;
    float4 identity = prefixOp.identity();
    float4 prev_result = identity;
    __shared__ float4 shared_prev_result;

    if(h0 != nullptr){
        c10::complex<T_ACC> h0_v = h0[(b * D + d) * N + n];
        prev_result = make_float4(1.f, 0.f, h0_v.real(), h0_v.imag());
    }
    float p_v = static_cast<float>(p[d * N + n]);
    c10::complex<float> q_v = static_cast<c10::complex<float>>(q[d * N + n]);
    c10::complex<float> gamma_v = static_cast<c10::complex<float>>(gamma[d * N + n]);

    for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
        // Save prev_result
        if (tid == 0) {
            int64_t idx = (((b * D + d) * N + n) * num_chunks + chunk);
            chunk_decay[idx] = c10::complex<T_ACC>(prev_result.x, prev_result.y);
            chunk_gain[idx] = c10::complex<T_ACC>(prev_result.z, prev_result.w);
        }
        __syncthreads();

        // Load
        int64_t start = chunk * CHUNK;
        float4 local_original = identity;
        if (start + tid < L){
            bool is_bos = (bos_mask == nullptr) ? 0 : bos_mask[b * L + start + tid];
            float x_v = static_cast<float>(x[(b * D + d) * L + (start + tid)]);
            float4 v = make_float4(is_bos ? 0.f : q_v.real(),
                                   is_bos ? 0.f : q_v.imag(),
                                   p_v * x_v,
                                   0.f);

            shared_result[tid] = v;
            local_original = v;
        }
        else {
            shared_result[tid] = identity;
        }
        __syncthreads();

        // Upsweep
        for (int64_t stride = 1; stride < CHUNK; stride <<= 1) {
            int64_t offset = stride << 1;
            int64_t idx = (tid + 1) * offset - 1;
            if (idx < CHUNK) {
                shared_result[idx] = prefixOp(shared_result[idx - stride], shared_result[idx]);
            }
            __syncthreads();
        }

        float4 chunk_total = identity;
        if (tid == CHUNK - 1) {
            chunk_total = shared_result[tid];
        }
        __syncthreads();

        // Zero root
        if (tid == CHUNK - 1) {
            shared_result[CHUNK - 1] = identity;
        }
        __syncthreads();

        // Downsweep
        for (int64_t d_str = CHUNK >> 1; d_str > 0; d_str >>= 1) {
            int64_t offset = d_str << 1;
            int64_t idx = (tid * offset) + offset - 1;
            if (idx < CHUNK) {
                float4 tmp = shared_result[idx - d_str];
                shared_result[idx - d_str] = shared_result[idx];
                shared_result[idx] = prefixOp(shared_result[idx], tmp);
            }
            __syncthreads();
        }

        // Write back to h
        if (start + tid < L) {
            int64_t t = start + tid;
            float4 local_h = prefixOp(shared_result[tid], local_original);
            float4 global_h = prefixOp(prev_result, local_h);

            if (t == L - 1) {
                h[(b * D + d) * N + n] = c10::complex<T_ACC>(global_h.z, global_h.w);
            }

            T_ACC contrib = static_cast<T_ACC>(global_h.z * gamma_v.real() - global_h.w * gamma_v.imag());
            gpuAtomicAdd(&y[(b * D + d) * L + t], contrib);
        }
        __syncthreads();

        // Update prev_result
        if (tid == CHUNK - 1) {
            prev_result = prefixOp(prev_result, chunk_total);
            shared_prev_result = prev_result;
        }
        __syncthreads();
        prev_result = shared_prev_result;
        __syncthreads();
    }
}


template <typename T, typename T_ACC, int64_t CHUNK=256>
__global__ void CEMABlellochScanCUDABwdKernel(int64_t D, int64_t N, int64_t L,
                                              const T_ACC* __restrict__ y_grad,
                                            //   const T* __restrict__ y_grad,
                                              const c10::complex<T_ACC>* __restrict__ h_last_grad,
                                              const c10::complex<T_ACC>* __restrict__ chunk_decay,
                                              const c10::complex<T_ACC>* __restrict__ chunk_gain,
                                              const T* __restrict__ x,
                                              const T_ACC* __restrict__ p,
                                              const c10::complex<T_ACC>* __restrict__ q,
                                              const c10::complex<T_ACC>* __restrict__ gamma,
                                              const bool* __restrict__ bos_mask,
                                            //   T* __restrict__ x_grad,
                                              T_ACC* __restrict__ x_grad,
                                              T_ACC* __restrict__ p_grad,
                                              c10::complex<T_ACC>* __restrict__ q_grad,
                                              c10::complex<T_ACC>* __restrict__ gamma_grad,
                                              c10::complex<T_ACC>* __restrict__ h0_grad) {
    const int64_t b = blockIdx.x;
    const int64_t d = blockIdx.y;
    const int64_t n = blockIdx.z;
    const int64_t tid = threadIdx.x;

    if (d >= D || n >= N) return;

    // const int64_t CHUNK = blockDim.x;
    const int64_t num_chunks = (L + CHUNK - 1) / CHUNK;

    extern __shared__ float4 bwd_scan_shm[];
    float4* shared_h_result = bwd_scan_shm;
    float4* shared_h_grad_result = bwd_scan_shm + CHUNK;

    // Init & load
    ComplexRightAffineProd prefixOp;
    ComplexLeftAffineProd suffixOp;

    float4 identity = suffixOp.identity();
    float4 next_grad_result = identity;
    __shared__ float4 shared_next_grad_result;

    float p_v = static_cast<float>(p[d * N + n]);
    c10::complex<float> q_v = static_cast<c10::complex<float>>(q[d * N + n]);
    c10::complex<float> gamma_v = static_cast<c10::complex<float>>(gamma[d * N + n]);

    float local_p_grad = 0.f;
    float local_q_grad_real = 0.f;
    float local_q_grad_imag = 0.f;
    float local_gamma_grad_real = 0.f;
    float local_gamma_grad_imag = 0.f;

    for (int64_t chunk = num_chunks - 1; chunk >= 0; --chunk) {
        // Scan h
        // Load
        int64_t start = chunk * CHUNK;
        float4 prev_result = identity;
        float4 local_h_original = identity;
        float4 local_h_grad_original = identity;
        if (start + tid < L) {
            bool is_bos = (bos_mask == nullptr) ? 0 : bos_mask[b * L + start + tid];
            float x_v = static_cast<float>(x[(b * D + d) * L + (start + tid)]);
            float4 h_v = make_float4(is_bos ? 0.f : q_v.real(),
                                     is_bos ? 0.f : q_v.imag(),
                                     p_v * x_v,
                                     0.f);
            shared_h_result[tid] = h_v;
            local_h_original = h_v;
            c10::complex<T_ACC> chunk_decay_v = chunk_decay[((b * D + d) * N + n) * num_chunks + chunk];
            c10::complex<T_ACC> chunk_gain_v = chunk_gain[((b * D + d) * N + n) * num_chunks + chunk];
            prev_result = make_float4(chunk_decay_v.real(), chunk_decay_v.imag(),
                                      chunk_gain_v.real(), chunk_gain_v.imag());
        }
        else {
            shared_h_result[tid] = identity;
        }
        __syncthreads();

        // Upsweep
        for (int64_t stride = 1; stride < CHUNK; stride <<= 1) {
            int64_t offset = stride << 1;
            int64_t idx = (tid + 1) * offset - 1;
            if (idx < CHUNK) {
                shared_h_result[idx] = prefixOp(shared_h_result[idx - stride], shared_h_result[idx]);
            }
            __syncthreads();
        }

        // Zero root
        if (tid == CHUNK - 1) {
            shared_h_result[CHUNK - 1] = identity;
        }
        __syncthreads();

        // Downsweep
        for (int64_t d_str = CHUNK >> 1; d_str > 0; d_str >>= 1) {
            int64_t offset = d_str << 1;
            int64_t idx = (tid * offset) + offset - 1;
            if (idx < CHUNK) {
                float4 tmp = shared_h_result[idx - d_str];
                shared_h_result[idx - d_str] = shared_h_result[idx];
                shared_h_result[idx] = prefixOp(shared_h_result[idx], tmp);
            }
            __syncthreads();
        }

        if (start + tid < L) {
            shared_h_result[tid] = prefixOp(prev_result, shared_h_result[tid]);
        }
        __syncthreads();

        // Scan h_grad
        // Load
        if (start + tid < L){
            bool next_is_bos = false;

            if (bos_mask != nullptr){
                if (start + tid != L - 1) next_is_bos = bos_mask[b * L + start + tid + 1];
                else next_is_bos = true;
            }

            float y_grad_v = static_cast<float>(y_grad[(b * D + d) * L + (start + tid)]);
            float h_last_grad_v_real = 0.f;
            float h_last_grad_v_imag = 0.f;
            if (h_last_grad != nullptr && start + tid == L - 1){
                c10::complex<T_ACC> h_last_grad_v = h_last_grad[(b * D + d) * N + n];
                h_last_grad_v_real = h_last_grad_v.real();
                h_last_grad_v_imag = h_last_grad_v.imag();
            }
            float4 h_grad_v = make_float4(next_is_bos ? 0.f : q_v.real(),
                                          next_is_bos ? 0.f : -q_v.imag(),
                                          y_grad_v * gamma_v.real() + h_last_grad_v_real,
                                          -y_grad_v * gamma_v.imag() + h_last_grad_v_imag);

            shared_h_grad_result[tid] = h_grad_v;
            local_h_grad_original = h_grad_v;
        }
        else {
            shared_h_grad_result[tid] = identity;
        }
        __syncthreads();

        // Upsweep
        for (int64_t stride = 1; stride < CHUNK; stride <<= 1) {
            int64_t offset = stride << 1;
            int64_t idx = CHUNK - (tid + 1) * offset;
            if (idx >= 0) {
                shared_h_grad_result[idx] = suffixOp(shared_h_grad_result[idx], shared_h_grad_result[idx + stride]);
            }
            __syncthreads();
        }

        // Save chunk total
        float4 chunk_total = identity;
        if (tid == 0) {
            chunk_total = shared_h_grad_result[0];
        }
        __syncthreads();

        // Zero root
        if (tid == 0) {
            shared_h_grad_result[0] = identity;
        }
        __syncthreads();

        // Downsweep
        for (int64_t d_str = CHUNK >> 1; d_str > 0; d_str >>= 1) {
            int64_t offset = d_str << 1;
            int64_t idx = CHUNK - (tid + 1) * offset;
            if (idx >= 0) {
                float4 tmp = shared_h_grad_result[idx + d_str];
                shared_h_grad_result[idx + d_str] = shared_h_grad_result[idx];
                shared_h_grad_result[idx] = suffixOp(tmp, shared_h_grad_result[idx]);
            }
            __syncthreads();
        }

        // Add next result
        if (start + tid < L) {
            int64_t t = start + tid;
            bool is_bos = false;
            if (bos_mask != nullptr) is_bos = bos_mask[b * L + t];

            float4 local_h_grad = suffixOp(local_h_grad_original, shared_h_grad_result[tid]);
            float4 global_h_grad = suffixOp(local_h_grad, next_grad_result);

            float grad_h_real = global_h_grad.z;
            float grad_h_imag = global_h_grad.w;

            float4 ht_prev = shared_h_result[tid];
            float4 ht = prefixOp(ht_prev, local_h_original);

            // x grad
            float x_grad_sub = p_v * grad_h_real;
            atomicAdd(&x_grad[(b * D + d) * L + t], x_grad_sub);

            // gpuAtomicAdd(&x_grad[(b * D + d) * L + t], x_grad_sub);
            // x_grad[(b * D + d) * L + t] = x_grad_sub;

            // p grad
            float x_v = static_cast<float>(x[(b * D + d) * L + t]);
            local_p_grad += x_v * grad_h_real;

            // q grad
            if (!is_bos){
                float q_grad_real = ht_prev.z * grad_h_real + ht_prev.w * grad_h_imag;
                float q_grad_imag = ht_prev.z * grad_h_imag - ht_prev.w * grad_h_real;

                local_q_grad_real += q_grad_real;
                local_q_grad_imag += q_grad_imag;
            }

            // gamma grad
            float y_grad_v = static_cast<float>(y_grad[(b * D + d) * L + (start + tid)]);
            float gamma_grad_real = ht.z * y_grad_v;
            float gamma_grad_imag = -ht.w * y_grad_v;

            local_gamma_grad_real += gamma_grad_real;
            local_gamma_grad_imag += gamma_grad_imag;

            // h0 grad
            if (chunk == 0 && tid == 0 && !is_bos) {
                float h0_grad_real = grad_h_real * q_v.real() + grad_h_imag * q_v.imag();
                float h0_grad_imag = grad_h_imag * q_v.real() - grad_h_real * q_v.imag();

                // gpuAtomicAdd(&h0_grad[(b * D + d) * N + n], c10::complex<T_ACC>(h0_grad_real, h0_grad_imag));
                h0_grad[(b * D + d) * N + n] = c10::complex<T_ACC>(h0_grad_real, h0_grad_imag);
            }
        }
        __syncthreads();

        // Update next result
        if (tid == 0) {
            next_grad_result = suffixOp(chunk_total, next_grad_result);
            shared_next_grad_result = next_grad_result;
        }
        __syncthreads();
        next_grad_result = shared_next_grad_result;
        __syncthreads();
    }

    atomicAdd(&p_grad[d * N + n], static_cast<T_ACC>(local_p_grad));
    gpuAtomicAdd(&q_grad[d * N + n], c10::complex<T_ACC>(local_q_grad_real, local_q_grad_imag));
    gpuAtomicAdd(&gamma_grad[d * N + n], c10::complex<T_ACC>(local_gamma_grad_real, local_gamma_grad_imag));
}

#define DISPATCH_SCAN_KERNEL(                                                \
    KernelFunc, T, T_ACC, shm, stream, B, D, N, L, ...)         \
  do {                                                                       \
    KernelFunc<T, T_ACC, kMaxNumThreads>                                                     \
        <<<dim3(B, D, N), kMaxNumThreads, shm, stream>>>(D, N, L, __VA_ARGS__); \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
  } while (false)

template <typename T>
void CEMABlellochScanCUDAFwdImpl(const torch::Tensor& x,
                                const torch::Tensor& p,
                                const torch::Tensor& q,
                                const torch::Tensor& gamma,
                                const torch::Tensor& bos_mask,
                                const torch::Tensor& h0,
                                torch::Tensor& h,
                                torch::Tensor& y,
                                torch::Tensor& chunk_decay,
                                torch::Tensor& chunk_gain) {
    using T_ACC = at::acc_type<T, true>;

    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t L = x.size(2);
    const int64_t N = p.size(1);

    const T* x_data = x.data_ptr<T>();
    const T_ACC* p_data = p.data_ptr<T_ACC>();
    const c10::complex<T_ACC>* q_data = q.data_ptr<c10::complex<T_ACC>>();
    const c10::complex<T_ACC>* gamma_data = gamma.data_ptr<c10::complex<T_ACC>>();

    const bool* bos_mask_data = bos_mask.defined() ? bos_mask.data_ptr<bool>() : nullptr;
    const c10::complex<T_ACC>* h0_data = h0.defined() ? h0.data_ptr<c10::complex<T_ACC>>() : nullptr;

    c10::complex<T_ACC>* h_data = h.data_ptr<c10::complex<T_ACC>>();
    T_ACC* y_data = y.data_ptr<T_ACC>();
    c10::complex<T_ACC>* chunk_decay_data = chunk_decay.data_ptr<c10::complex<T_ACC>>();
    c10::complex<T_ACC>* chunk_gain_data = chunk_gain.data_ptr<c10::complex<T_ACC>>();

    at::cuda::OptionalCUDAGuard guard(at::device_of(x));
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

    // const int64_t num_threads = cuda_utils::RowwiseNumThreads(L, kMaxNumThreads);

    const size_t shm_size = kMaxNumThreads * sizeof(float4);

    DISPATCH_SCAN_KERNEL(
        CEMABlellochScanCUDAFwdKernel, T, T_ACC,
        shm_size, cuda_stream,
        B, D, N, L, /*num_threads,*/
        x_data, p_data, q_data, gamma_data,
        bos_mask_data, h0_data,
        h_data, y_data, chunk_decay_data, chunk_gain_data);
}

template <typename T>
void CEMABlellochScanCUDABwdImpl(const torch::Tensor& y_grad,
                                const torch::Tensor& h_last_grad,
                                const torch::Tensor& chunk_decay,
                                const torch::Tensor& chunk_gain,
                                const torch::Tensor& x,
                                const torch::Tensor& p,
                                const torch::Tensor& q,
                                const torch::Tensor& gamma,
                                const torch::Tensor& bos_mask,
                                torch::Tensor& x_grad,
                                torch::Tensor& p_grad,
                                torch::Tensor& q_grad,
                                torch::Tensor& gamma_grad,
                                torch::Tensor& h0_grad) {
    using T_ACC = at::acc_type<T, true>;

    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t L = x.size(2);
    const int64_t N = p.size(1);

    const T_ACC* y_grad_data = y_grad.data_ptr<T_ACC>();
    // const T* y_grad_data = y_grad.data_ptr<T>();
    const c10::complex<T_ACC>* chunk_decay_data = chunk_decay.data_ptr<c10::complex<T_ACC>>();
    const c10::complex<T_ACC>* chunk_gain_data = chunk_gain.data_ptr<c10::complex<T_ACC>>();
    const T* x_data = x.data_ptr<T>();
    const T_ACC* p_data = p.data_ptr<T_ACC>();
    const c10::complex<T_ACC>* q_data = q.data_ptr<c10::complex<T_ACC>>();
    const c10::complex<T_ACC>* gamma_data = gamma.data_ptr<c10::complex<T_ACC>>();

    const bool* bos_mask_data = bos_mask.defined() ? bos_mask.data_ptr<bool>() : nullptr;
    const c10::complex<T_ACC>* h_last_grad_data =
        h_last_grad.defined() ? h_last_grad.data_ptr<c10::complex<T_ACC>>() : nullptr;

    // T* x_grad_data = x_grad.data_ptr<T>();
    T_ACC* x_grad_data = x_grad.data_ptr<T_ACC>();
    T_ACC* p_grad_data = p_grad.data_ptr<T_ACC>();
    c10::complex<T_ACC>* q_grad_data = q_grad.data_ptr<c10::complex<T_ACC>>();
    c10::complex<T_ACC>* gamma_grad_data = gamma_grad.data_ptr<c10::complex<T_ACC>>();
    c10::complex<T_ACC>* h0_grad_data = h0_grad.data_ptr<c10::complex<T_ACC>>();

    at::cuda::OptionalCUDAGuard guard(at::device_of(x));
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

    // const int64_t num_threads = cuda_utils::RowwiseNumThreads(L, kMaxNumThreads);

    const size_t shm_size = 2 * kMaxNumThreads * sizeof(float4);

    DISPATCH_SCAN_KERNEL(
        CEMABlellochScanCUDABwdKernel, T, T_ACC,
        shm_size, cuda_stream,
        B, D, N, L, /*num_threads,*/
        y_grad_data, h_last_grad_data,
        chunk_decay_data, chunk_gain_data,
        x_data, p_data, q_data, gamma_data,
        bos_mask_data,
        x_grad_data, p_grad_data, q_grad_data, gamma_grad_data, h0_grad_data);
}

#undef DISPATCH_SCAN_KERNEL
} // namespace

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor> CEMABlellochScanCUDAFwd(const torch::Tensor& x,
                                                                const torch::Tensor& p,
                                                                const torch::Tensor& q,
                                                                const torch::Tensor& gamma,
                                                                const c10::optional<torch::Tensor>& bos_mask,
                                                                const c10::optional<torch::Tensor>& h0) {
    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t L = x.size(2);
    const int64_t N = p.size(1);

    // const int64_t chunk_size = cuda_utils::RowwiseNumThreads(L, kMaxNumThreads);
    const int64_t chunk_size = kMaxNumThreads;
    const int64_t num_chunks = (L + chunk_size - 1) / chunk_size;

    c10::MaybeOwned<torch::Tensor> bos_mask_maybe_owned =
        at::borrow_from_optional_tensor(bos_mask);
    c10::MaybeOwned<torch::Tensor> h0_maybe_owned =
        at::borrow_from_optional_tensor(h0);
    torch::Tensor h = torch::empty(
        {B, D, N}, q.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor y_acc = torch::zeros(
        {B, D, L}, p.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor chunk_decay = torch::empty(
        {B, D, N, num_chunks}, q.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor chunk_gain = torch::empty_like(chunk_decay);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "CEMABlellochScanCUDAFwd", [&]() {
        CEMABlellochScanCUDAFwdImpl<scalar_t>(*(x.expect_contiguous()),
            *(p.expect_contiguous()), *(q.expect_contiguous()), *(gamma.expect_contiguous()),
            *(bos_mask_maybe_owned->expect_contiguous()), *(h0_maybe_owned->expect_contiguous()),
            h, y_acc, chunk_decay, chunk_gain);
    });
    torch::Tensor y = y_acc.to(x.scalar_type());
    return std::make_tuple<torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor>(std::move(y),
                                                         std::move(h),
                                                         std::move(chunk_decay),
                                                         std::move(chunk_gain));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor> CEMABlellochScanCUDABwd(const torch::Tensor& y_grad,
                                                                const c10::optional<torch::Tensor>& h_last_grad,
                                                                const torch::Tensor& chunk_decay,
                                                                const torch::Tensor& chunk_gain,
                                                                const torch::Tensor& x,
                                                                const torch::Tensor& p,
                                                                const torch::Tensor& q,
                                                                const torch::Tensor& gamma,
                                                                const c10::optional<torch::Tensor>& bos_mask){
    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t L = x.size(2);
    const int64_t N = p.size(1);

    torch::Tensor y_grad_acc = y_grad.to(p.scalar_type());

    c10::MaybeOwned<torch::Tensor> bos_mask_maybe_owned =
        at::borrow_from_optional_tensor(bos_mask);
    c10::MaybeOwned<torch::Tensor> h_last_grad_maybe_owned =
        at::borrow_from_optional_tensor(h_last_grad);

    torch::Tensor x_grad_acc = torch::zeros(
        {B, D, L}, p.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor p_grad = torch::zeros(
        {D, N}, p.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor q_grad = torch::zeros(
        {D, N}, q.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor gamma_grad = torch::zeros(
        {D, N}, gamma.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor h0_grad = torch::zeros(
        {B, D, N}, q.options().memory_format(at::MemoryFormat::Contiguous));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "CEMABlellochScanCUDABwd", [&]() {
        CEMABlellochScanCUDABwdImpl<scalar_t>(*(y_grad_acc.expect_contiguous()),
            *(h_last_grad_maybe_owned->expect_contiguous()),
            *(chunk_decay.expect_contiguous()), *(chunk_gain.expect_contiguous()),
            *(x.expect_contiguous()), *(p.expect_contiguous()), *(q.expect_contiguous()), *(gamma.expect_contiguous()),
            *(bos_mask_maybe_owned->expect_contiguous()),
            x_grad_acc, p_grad, q_grad, gamma_grad, h0_grad);
    });

    torch::Tensor x_grad = x_grad_acc.to(x.scalar_type());

    return std::make_tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor>(std::move(x_grad),
                                                         std::move(p_grad),
                                                         std::move(q_grad),
                                                         std::move(gamma_grad),
                                                         std::move(h0_grad));
}

} // namespace ops
} // namespace gecko
