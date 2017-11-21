#include "./bilinear_pool-inl.h"

#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

namespace mshadow {
} // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(BilinearPoolParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BilinearPoolOp<cpu, DType>(param);
  });
  return op;
}

Operator *BilinearPoolProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(BilinearPoolParam);
MXNET_REGISTER_OP_PROPERTY(_contrib_BilinearPool, BilinearPoolProp)
.describe("bilinear pool.")
.add_argument("data0", "NDArray-or-Symbol", "data0")
.add_argument("data1", "NDArray-or-Symbol", "data1")
.add_arguments(BilinearPoolParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
