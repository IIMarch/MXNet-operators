#include "./bilinear_pool-inl.h"

#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>


namespace mshadow {
namespace cuda {

}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(BilinearPoolParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BilinearPoolOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
