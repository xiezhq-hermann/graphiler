#include "bcast.h"
#include <torch/torch.h>

namespace dgl {

bool UseBcast(const std::string &op, at::Tensor lhs, at::Tensor rhs) {
  if (op == "copy_lhs" || op == "copy_rhs")
    return false; // broadcasting is not required for copy_u/copy_e
  if (lhs.ndimension() != rhs.ndimension())
    return true;
  for (int i = 1; i < lhs.ndimension(); ++i) {
    if (lhs.sizes()[i] != rhs.sizes()[i])
      return true;
  }
  return false;
}

BcastOff CalcBcastOff(const std::string &op, at::Tensor lhs, at::Tensor rhs) {
  auto lhs_shp = lhs.sizes(), rhs_shp = rhs.sizes();
  auto lhs_ndim = lhs.ndimension(), rhs_ndim = rhs.ndimension();
  std::vector<int64_t> lhs_off, rhs_off;
  lhs_off.push_back(0);
  rhs_off.push_back(0);
  int64_t lhs_len = 1, rhs_len = 1;
  for (int i = 1; i < lhs_shp.size(); ++i) {
    lhs_len *= lhs_shp[i];
  }
  for (int i = 1; i < rhs_shp.size(); ++i) {
    rhs_len *= rhs_shp[i];
  }
  bool use_bcast = UseBcast(op, lhs, rhs);
  int64_t reduce_size = 1; // defaults to 1
  int64_t out_len = 1;
  if (use_bcast) {
    const int max_ndim = std::max(lhs_ndim, rhs_ndim) - 1;
    int64_t j = 0;
    if (op == "dot") {
      reduce_size = lhs.sizes()[lhs_ndim - 1];
      ++j;
    }
    int64_t stride_l = 1, stride_r = 1;
    for (; j < max_ndim; ++j) {
      const int64_t dl = (lhs_ndim - 1 - j < 1) ? 1 : lhs_shp[lhs_ndim - 1 - j];
      const int64_t dr = (rhs_ndim - 1 - j < 1) ? 1 : rhs_shp[rhs_ndim - 1 - j];
      for (int i = 1; i < std::max(dl, dr); ++i) {
        for (int k = 0; k < out_len; ++k) {
          lhs_off.push_back(lhs_off[k] + i * (i < dl) * stride_l);
          rhs_off.push_back(rhs_off[k] + i * (i < dr) * stride_r);
        }
      }
      out_len *= std::max(dl, dr);
      stride_l *= dl;
      stride_r *= dr;
    }
  } else {
    out_len = (op == "copy_rhs") ? rhs_len : lhs_len;
    if (op == "dot") {
      reduce_size = lhs_shp[lhs_ndim - 1];
      out_len /= reduce_size;
    }
  }

  auto opts = at::TensorOptions(torch::kInt64).device(lhs.device());
  at::Tensor lhs_off_tensor =
      torch::from_blob(lhs_off.data(), {lhs_len}).to(opts);
  at::Tensor rhs_off_tensor =
      torch::from_blob(rhs_off.data(), {rhs_len}).to(opts);

  return BcastOff({lhs_off_tensor, rhs_off_tensor, use_bcast, lhs_len, rhs_len,
                   out_len, reduce_size});
}

} // namespace dgl