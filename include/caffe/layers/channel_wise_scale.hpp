#ifndef CAFFE_CHANNEL_WISE_SCALE_HPP_
#define CAFFE_CHANNEL_WISE_SCALE_HPP_
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe{
template<typename Dtype>
class ChannelWiseScaleLayer : public Layer<Dtype> {
 public:
  explicit ChannelWiseScaleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "ChannelWiseScale"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void FeatureMapBackward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void ScaleFeatureBackward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline Dtype get_sum(Dtype *array,int size);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int num_;
  int channels_;
  int height_;
  int width_;
};
}
#endif  // CAFFE_CHANNEL_WISE_SCALE_HPP_
