#include <torch/torch.h>

#include "ops/attention.h"
#include "ops/attention_softmax.h"
#include "ops/ema_hidden.h"
#include "ops/ema_parameters.h"
#include "ops/cema_blelloch_scan.h"
#include "ops/cema_cub_scan.h"
#include "ops/fftconv.h"
#include "ops/group_rms_norm.h"
#include "ops/group_layer_norm.h"
#include "ops/sequence_norm.h"
#include "ops/timestep_norm.h"
#include "utils.h"

namespace gecko {

PYBIND11_MODULE(gecko_extension, m) {
  m.doc() = "Gecko Cpp Extensions.";
  py::module m_ops = m.def_submodule("ops", "Submodule for custom ops.");
  ops::DefineAttentionOp(m_ops);
  ops::DefineMultiSegAttentionOp(m_ops);
  ops::DefineAttentionSoftmaxOp(m_ops);
  ops::DefineEMAHiddenOp(m_ops);
  ops::DefineEMAParametersOp(m_ops);
  ops::DefineCEMABlellochScanOp(m_ops);
  ops::DefineCEMACUBScanOp(m_ops);
  ops::DefineFFTConvOp(m_ops);
  ops::DefineGroupLayerNormOp(m_ops);
  ops::DefineGroupRMSNormOp(m_ops);
  ops::DefineSequenceNormOp(m_ops);
  ops::DefineTimestepNormOp(m_ops);
}

}  // namespace gecko
