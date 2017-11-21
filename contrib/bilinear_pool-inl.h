/* !
 * Copyright (c) 2017 by Contributors
 * \file bilinear-pool-inl.h
 * \brief
 * \author Kang Yang
*/

#ifndef MXNET_OPERATOR_BILINEAR_POOL_INL_H_
#define MXNET_OPERATOR_BILINEAR_POOL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {

namespace bilinear_pool {
enum BilinearPoolOpInputs {kData0, kData1};
enum BilinearPoolOpOutputs {kOut};
}

struct BilinearPoolParam : public dmlc::Parameter<BilinearPoolParam> {
  DMLC_DECLARE_PARAMETER(BilinearPoolParam) {
  }
};


template<typename xpu, typename DType>
class BilinearPoolOp : public Operator {
  public:
    explicit BilinearPoolOp(BilinearPoolParam p) {
      this->param_ = p;
    }

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      CHECK_EQ(in_data.size(), 2);
      CHECK_EQ(out_data.size(), 1);
      for (int i = 0; i < 2; ++i) {
        CHECK_EQ(in_data[i].ndim(), 4);
      }
      for (int axis = 0; axis < 4; ++axis) {
        if (axis == 1)
          continue;
        CHECK_EQ(in_data[bilinear_pool::kData0].shape_[axis],
                 in_data[bilinear_pool::kData1].shape_[axis]);
      }

      Stream<xpu> *s = ctx.get_stream<xpu>();
      Tensor<xpu, 4, DType> data0 = in_data[bilinear_pool::kData0].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> data1 = in_data[bilinear_pool::kData1].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = out_data[bilinear_pool::kOut].get<xpu, 4, DType>(s);

      CHECK_EQ(data0.CheckContiguous(), true);
      CHECK_EQ(data1.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);

      out = -FLT_MAX;

      const int batch_size = data0.size(0);
      const int height = data0.size(2);
      const int width = data0.size(3);

      DType *data0_ptr = data0.dptr_;
      DType *data1_ptr = data1.dptr_;
      DType *out_ptr = out.dptr_;

      const int step_out = out.template MemSize<1>();
      const int step_data_0 = data0.template MemSize<1>();
      const int step_data_1 = data1.template MemSize<1>();

      for (int b = 0; b < batch_size; ++b) {
        Tensor<xpu, 2, DType> d0(data0_ptr + b * step_data_0, Shape2(data0.size(1), height*width), s);
        Tensor<xpu, 2, DType> d1(data1_ptr + b * step_data_1, Shape2(data1.size(1), height*width), s);
        Tensor<xpu, 2, DType> o(out_ptr + b * step_out, Shape2(data0.size(1), data1.size(1)), s);
        o = dot(d0, d1.T());
      }
    }

    virtual void Backward(const OpContext &ctx,
                          const std::vector<TBlob> &out_grad,
                          const std::vector<TBlob> &in_data,
                          const std::vector<TBlob> &out_data,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &in_grad,
                          const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      CHECK_EQ(in_grad.size(), 2);
      CHECK_EQ(out_grad.size(), 1);
      
      Stream<xpu> *s = ctx.get_stream<xpu>();
      Tensor<xpu, 4, DType> data0 = in_data[bilinear_pool::kData0].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> data1 = in_data[bilinear_pool::kData1].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = out_data[bilinear_pool::kOut].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> grad_out = out_grad[bilinear_pool::kOut].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> grad_data0 = in_grad[bilinear_pool::kData0].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> grad_data1 = in_grad[bilinear_pool::kData1].get<xpu, 4, DType>(s);

      CHECK_EQ(data0.CheckContiguous(), true);
      CHECK_EQ(data1.CheckContiguous(), true);
      CHECK_EQ(grad_out.CheckContiguous(), true);
      CHECK_EQ(grad_data0.CheckContiguous(), true);
      CHECK_EQ(grad_data1.CheckContiguous(), true);

      const int batch_size = data0.size(0);
      const int height = data0.size(2);
      const int width = data0.size(3);

      const int c0 = data0.size(1);
      const int c1 = data1.size(1);
      
      grad_data0 = 0;
      grad_data1 = 0;

      DType* top_diff = grad_out.dptr_;
      DType* bottom_diff[2] = { grad_data0.dptr_, grad_data1.dptr_ };
      DType* bottom_data[2] = { data0.dptr_, data1.dptr_ };

      const int step_top = out.template MemSize<1>();
      const int step_data_0 = data0.template MemSize<1>();
      const int step_data_1 = data1.template MemSize<1>();
      const int a = out.size(1) * out.size(2) * out.size(3);
      const int b = data0.size(1) * data1.size(2) * data1.size(3);
      const int c = data1.size(1) * data1.size(2) * data1.size(3);
      CHECK_EQ(step_top, a);
      CHECK_EQ(step_data_0, b);
      CHECK_EQ(step_data_1, c);

      const int step_bottom[2] = { step_data_0, step_data_1 };
      const int c_bottom[2] = { c0, c1 };

      for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < 2; ++i) {
          if ((bottom_data[0] == bottom_data[1]) && (i == 1)) {
            Tensor<xpu, 2, DType> g0(bottom_diff[0] + b * step_bottom[0], Shape2(data0.size(1), height*width), s);
            g0 *= 2.0;
            continue;
          }

          bool isTrans = ((i == 0) ? false : true);
          Tensor<xpu, 2, DType> go(top_diff + b * step_top, Shape2(data0.size(1), data1.size(1)), s);
          Tensor<xpu, 2, DType> di(bottom_data[1 - i] + b * step_bottom[1 - i], Shape2(c_bottom[1 - i], height*width), s);
          Tensor<xpu, 2, DType> gi(bottom_diff[i] + b * step_bottom[i], Shape2(c_bottom[i], height*width), s);
          
          if (isTrans) {
            gi = dot(go.T(), di);
          } else {
            gi = dot(go, di);
          }
        }
      }
    }
  private:
    BilinearPoolParam param_;

};

template<typename xpu>
Operator* CreateOp(BilinearPoolParam param, int dtype);


#if DMLC_USE_CXX11
class BilinearPoolProp : public OperatorProperty {
  public:
    std::vector<std::string> ListArguments() const override {
      return {"data0", "data1"};
    }

    std::vector<std::string> ListOutputs() const override {
      return {"output"};
    }

    int NumOutputs() const override {
      return 1;
    }

    int NumVisibleOutputs() const override {
      return 1;
    }

    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
      param_.Init(kwargs);
    }

    std::map<std::string, std::string> GetParams() const override {
      return param_.__DICT__();
    }

    bool InferShape(std::vector<TShape> *in_shape,
                    std::vector<TShape> *out_shape,
                    std::vector<TShape> *aux_shape) const override {
      using namespace mshadow;
      CHECK_EQ(in_shape->size(), 2U) << "Input:[data0, data1]";
      TShape d0shape = in_shape->at(bilinear_pool::kData0);
      CHECK_EQ(d0shape.ndim(), 4U) << "data should be a 4D tensor";
      TShape d1shape = in_shape->at(bilinear_pool::kData1);
      CHECK_EQ(d1shape.ndim(), 4U) << "data should be a 4D tensor";

      for (int axis = 0; axis < 4; ++axis) {
        if (axis == 1)
          continue;
        CHECK_EQ(d0shape[axis], d1shape[axis]) << "Two bottom blobs not compatible at axis " << axis << ".";
      }

      out_shape->clear();
      out_shape->push_back(
          Shape4(d0shape[0], d0shape[1]*d1shape[1], 1, 1));
      return true;
    }

    bool InferType(std::vector<int> *in_type,
                   std::vector<int> *out_type,
                   std::vector<int> *aux_type) const override {
      CHECK_EQ(in_type->size(), 2U);
      int dtype = (*in_type)[0];
      CHECK_EQ(dtype, (*in_type)[1]);
      CHECK_NE(dtype, -1) << "Input must have specified type";

      out_type->clear();
      out_type->push_back(dtype);
      return true;
    }

    OperatorProperty* Copy() const override {
      BilinearPoolProp* bilinear_pool_sym = new BilinearPoolProp();
      bilinear_pool_sym->param_ = this->param_;
      return bilinear_pool_sym;
    }

    std::string TypeString() const override {
      return "_contrib_BilinearPool";
    }

    std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
      return {out_grad[bilinear_pool::kOut], 
        in_data[bilinear_pool::kData0], in_data[bilinear_pool::kData1], out_data[bilinear_pool::kOut]};
    }

    Operator* CreateOperator(Context ctx) const override {
      LOG(FATAL) << "Not Implemented.";
      return NULL;
    }
    Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                               std::vector<int> *in_type) const override;

  private:
    BilinearPoolParam param_;
};
#endif // DMLC_USE_CXX11



} // namespace op
} // namespace mxnet



#endif // MXNET_OPERATOR_BILINEAR_POOL_INL_H_


