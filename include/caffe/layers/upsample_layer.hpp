#ifndef CAFFE_UPSAMPLE_LAYERS_HPP_
#define CAFFE_UPSAMPLE_LAYERS_HPP_
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class UpsamplePoolingLayer : public Layer<Dtype> {
 public:
  explicit UpsamplePoolingLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "UpsamplePooling"; }

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void bilinear_interpolate(const Dtype* bottom_data, const int height, 
    const int width, Dtype h, Dtype w, Dtype & value);
  virtual Dtype get_feature_gradient(Dtype argmax_h, Dtype argmax_w, const int h,
      const int w, const int height, const int width);
  Dtype spatial_scale_;

  int channels_;
  int height_;
  int width_;
 // int num_;

  int pooled_height_;
  int pooled_width_;
};
}  // namespace caffe
#endif  // CAFFE_UPSAMPLE_LAYERS_HPP_