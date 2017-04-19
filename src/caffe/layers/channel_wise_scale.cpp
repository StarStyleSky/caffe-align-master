#include <cfloat>
#include <vector>

#include "caffe/layers/channel_wise_scale.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ChannelWiseScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //// This layer do not need any paramter
}
template <typename Dtype>
void ChannelWiseScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height(); 
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(),bottom[1]->channels())
  <<"feature map channel and scale feature map channel must be the same";
  top[0]->Reshape(num_,channels_,height_,width_);
}
template <typename Dtype>
void ChannelWiseScaleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

  const Dtype *scale_data = bottom[1]->cpu_data();
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data(); 

  caffe_set(top[0]->count(),Dtype(0.),top_data);

  int area = height_ * width_;
  for(int n = 0;n < num_; ++n){
    for(int c = 0;c < channels_; ++c){
      int offset = bottom[0]->offset(n,c);
      const Dtype *bottom_batch_data = bottom_data + offset;
      Dtype *top_batch_data = top_data + offset;
      Dtype scale = *(scale_data + bottom[1]->offset(n,c));
      caffe_cpu_scale(area,scale,bottom_batch_data,top_batch_data);
    }
  }
}
template <typename Dtype>
void ChannelWiseScaleLayer<Dtype>::FeatureMapBackward(const vector<Blob<Dtype>*>& top,
      const vector<Blob<Dtype>*>& bottom){

  const Dtype *top_diff = top[0]->cpu_diff();
  const Dtype *scale_data = bottom[1]->cpu_data();
  Dtype *feature_diff = bottom[0]->mutable_cpu_diff();

  caffe_set(bottom[0]->count(),Dtype(0.),feature_diff);

  int area = height_ * width_;
  for(int n = 0; n < num_ ; ++n){
    for(int c = 0;c < channels_ ; ++ c){
      int offset = bottom[0]->offset(n,c);
      const Dtype *top_batch_diff = top_diff + offset;
      Dtype *feature_batch_diff = feature_diff + offset;
      Dtype scale = *(scale_data + bottom[1]->offset(n,c));     
      caffe_cpu_scale(area,scale,top_batch_diff,feature_batch_diff);
    }
  }
}
template<typename Dtype>
Dtype ChannelWiseScaleLayer<Dtype>::get_sum(Dtype *array,int size){
  Dtype sum = 0.;
  for(int i = 0;i<size;++i)
    sum += array[i];
  return sum;
}
template<typename Dtype>
void ChannelWiseScaleLayer<Dtype>::ScaleFeatureBackward(const vector<Blob<Dtype>*>& top,
      const vector<Blob<Dtype>*>& bottom){

  const Dtype *top_diff = top[0]->cpu_diff();
  const Dtype *feature_map_data = bottom[1]->cpu_data();
  Dtype *scale_diff = bottom[1]->mutable_cpu_diff();
  caffe_set(bottom[1]->count(),Dtype(0.),scale_diff);

  int area = height_ * width_;

  Dtype *buffer = new Dtype[area];
  caffe_set(area,Dtype(1.),buffer);
  for(int n = 0;n<num_;++n){
    for(int c = 0; c < channels_; ++ c){
      int offset = bottom[0]->offset(n,c);
      const Dtype *top_batch_diff = top_diff + offset;
      const Dtype *feature_batch_diff = feature_map_data + offset;
      caffe_mul(area,top_batch_diff,feature_batch_diff,buffer);
      int index = bottom[1]->offset(n,c);
      scale_diff[index] = get_sum(buffer,area);
    }
  }
  delete buffer;
}
template <typename Dtype>
void ChannelWiseScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if(propagate_down[0]){
    FeatureMapBackward(top,bottom);
  }
  if(propagate_down[1]){
    ScaleFeatureBackward(top,bottom);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ChannelWiseScaleLayer);
#endif

INSTANTIATE_CLASS(ChannelWiseScaleLayer);
REGISTER_LAYER_CLASS(ChannelWiseScale);

}  // namespace caffe
