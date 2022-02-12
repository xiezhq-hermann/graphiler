#ifndef DGL_MACRO_CUH_
#define DGL_MACRO_CUH_

namespace dgl {

#define BCAST_SWITCH(use_bcast, UseBcast, ...)                                 \
  do {                                                                         \
    if ((use_bcast)) {                                                         \
      constexpr bool UseBcast = true;                                          \
      { __VA_ARGS__ }                                                          \
    } else {                                                                   \
      constexpr bool UseBcast = false;                                         \
      { __VA_ARGS__ }                                                          \
    }                                                                          \
  } while (0);

} // namespace dgl

#endif // DGL_MACRO_CUH_