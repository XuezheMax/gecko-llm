#include "ops/cema_blelloch_scan.h"

namespace gecko{
namespace ops{

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
CEMABlellochScanFwd(
    const torch::Tensor& x,
    const torch::Tensor& p,
    const torch::Tensor& q,
    const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& bos_mask,
    const c10::optional<torch::Tensor>& h0){
  return CEMABlellochScanCUDAFwd(x, p, q, gamma, bos_mask, h0);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
CEMABlellochScanBwd(
    const torch::Tensor& y_grad,
    const c10::optional<torch::Tensor>& h_last_grad,
    const torch::Tensor& chunk_q_prefix,
    const torch::Tensor& chunk_h_prefix,
    const torch::Tensor& x,
    const torch::Tensor& p,
    const torch::Tensor& q,
    const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& bos_mask){
  return CEMABlellochScanCUDABwd(y_grad, h_last_grad, chunk_q_prefix, chunk_h_prefix, x, p, q, gamma, bos_mask);
}

void DefineCEMABlellochScanOp(py::module& m) {
    m.def("cema_blelloch_scan_fwd", &CEMABlellochScanFwd, "CEMABlellochScanFwd")
        .def("cema_blelloch_scan_bwd", &CEMABlellochScanBwd, "CEMABlellochScanBwd");
}

}  // namespace ops
}  // namespace gecko
