#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

#include <tuple>

namespace gecko{
namespace ops{

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CEMABlellochScanFwd(
    const torch::Tensor& x,
    const torch::Tensor& p,
    const torch::Tensor& q,
    const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& bos_mask,
    const c10::optional<torch::Tensor>& h0);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CEMABlellochScanCUDAFwd(
    const torch::Tensor& x,
    const torch::Tensor& p,
    const torch::Tensor& q,
    const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& bos_mask,
    const c10::optional<torch::Tensor>& h0);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CEMABlellochScanBwd(
    const torch::Tensor& y_grad,
    const c10::optional<torch::Tensor>& h_last_grad,
    const torch::Tensor& chunk_q_prefix,
    const torch::Tensor& chunk_h_prefix,
    const torch::Tensor& x,
    const torch::Tensor& p,
    const torch::Tensor& q,
    const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& bos_mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CEMABlellochScanCUDABwd(
    const torch::Tensor& y_grad,
    const c10::optional<torch::Tensor>& h_last_grad,
    const torch::Tensor& chunk_q_prefix,
    const torch::Tensor& chunk_h_prefix,
    const torch::Tensor& x,
    const torch::Tensor& p,
    const torch::Tensor& q,
    const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& bos_mask);

void DefineCEMABlellochScanOp(py::module& m);

}  // namespace ops
}  // namespace gecko
