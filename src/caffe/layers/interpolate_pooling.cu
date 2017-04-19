#include <cfloat>
#include "caffe/fast_rcnn_layers.hpp"
using std::max;
using std::min;
namespace caffe {
template <typename Dtype>
__device__ void bilinear_interpolate(const Dtype* bottom_data, const int height, const int width, Dtype h, Dtype w, Dtype & value) {
  // deal with cases that inverse elements are out of feature map boundary
  if (h < -0.5 || h > height - 0.5 || w < -0.5 || w > width - 0.5) {
    //empty
    return;
  }
  
  if (h <= 0) h = 0;
  if (w <= 0) w = 0;
  
  int h_low = (int) h;
  int w_low = (int) w;
  int h_high;
  int w_high;
  
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = (Dtype) h_low;
  } else {
    h_high = h_low + 1;
  }
  
  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = (Dtype) w_low;
  } else {
    w_high = w_low + 1;
  }
  
  Dtype lh = h - h_low;
  Dtype lw = w - w_low;
  Dtype hh = 1 - lh, hw = 1 - lw;
  // do bilinear interpolation
  Dtype v1 = bottom_data[h_low * width + w_low];
  Dtype v2 = bottom_data[h_low * width + w_high];
  Dtype v3 = bottom_data[h_high * width + w_low];
  Dtype v4 = bottom_data[h_high * width + w_high];
  Dtype w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw; 
  value = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}
template <typename Dtype>
__global__ void InterpolateForward(const int nthreads, const Dtype* bottom_data, const int channels, 
            const int height, const int width,const int pooled_height, const int pooled_width,
            Dtype* top_data){
  CUDA_KERNEL_LOOP(index, nthreads){
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    Dtype value = 0;

    Dtype index_h = (ph + 0.5) * height/pooled_height - 0.5;
    Dtype index_w = (pw + 0.5) * height/pooled_height - 0.5;

    bottom_data += (n * channels + c) * height * width;
    bilinear_interpolate(bottom_data,height,width,index_h,index_w,value);
    top_data[index] = value;
  }
}
template <typename Dtype>
void InterpolateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  InterpolateForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
    (count, bottom_data, channels_, height_, width_, pooled_height_,pooled_width_, top_data);
  CUDA_POST_KERNEL_CHECK;
}
template<typename Dtype>
__device__ Dtype get_feature_gradient(Dtype argmax_h, Dtype argmax_w, const int h,
      const int w, const int height, const int width){
  if (argmax_h < -0.5 || argmax_h >(height - 0.5) || argmax_w < -0.5 || argmax_w >(width - 0.5)){
      //empty
      return 0;
  }
  if (argmax_h < 0) argmax_h = 0;
  if (argmax_w < 0) argmax_w = 0;
  
  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (Dtype)argmax_h_low;
  }
  else
    argmax_h_high = argmax_h_low + 1;
  
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (Dtype)argmax_w_low;
  }
  else
    argmax_w_high = argmax_w_low + 1;
  
  Dtype weight = 0;
  if (h == argmax_h_low) {
    if (w == argmax_w_low) {
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    }
    else if (w == argmax_w_high) {
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    }
  }
  else if (h == argmax_h_high) {
    if (w == argmax_w_low) {
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    }
    else if (w == argmax_w_high) {
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    }
  }
  return weight;
}    
template <typename Dtype>
__global__ void InterpolateBackwardFeature(const int nthreads, const Dtype* top_diff,
          const int channels,const int height, const int width, const int pooled_height,
          const int pooled_width, Dtype* bottom_diff){
  CUDA_KERNEL_LOOP(index, nthreads){
    // (n,c,ph,pw) in top[0] feature map
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    Dtype weight_upper_left,weight_upper_right,weight_lower_left,weight_lower_right;

    Dtype index_h = (ph + 0.5) * height / pooled_height - 0.5;
    Dtype index_w = (pw + 0.5) * width / pooled_width - 0.5;

    weight_upper_left = get_feature_gradient(index_h,index_w,(int)index_h,(int)index_w,height,width);
    weight_upper_right = get_feature_gradient(index_h,index_w,(int)index_h,(int)index_w+1,height,width);
    weight_lower_left = get_feature_gradient(index_h,index_w,(int)index_h+1,(int)index_w,height,width);
    weight_lower_right = get_feature_gradient(index_h,index_w,(int)index_h+1,(int)index_w+1,height,width);

    const int offset_upper_left = ((n * channels + c) * height + (int)index_h) * width + (int)index_w;
    const int offset_upper_right = ((n * channels + c) * height + (int)index_h) * width + (int)index_w + 1;
    const int offset_lower_left = ((n * channels + c) * height + (int)index_h + 1) * width +(int)index_w;
    const int offset_lower_right = ((n * channels + c) * height + (int)index_h + 1) * width +(int)index_w + 1;

    bottom_diff[offset_upper_left] += weight_upper_left * top_diff[index];
    bottom_diff[offset_upper_right] += weight_upper_right * top_diff[index];
    bottom_diff[offset_lower_left] += weight_lower_left * top_diff[index];
    bottom_diff[offset_lower_right] += weight_lower_right * top_diff[index];
  }
}
template <typename Dtype>
void InterpolateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int count = top[0]->count();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  if(propagate_down[0]){
    InterpolateBackwardFeature<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
    (count, top_diff, channels_, height_, width_, pooled_height_,pooled_width_, bottom_diff);
  }
  CUDA_POST_KERNEL_CHECK;
}
INSTANTIATE_LAYER_GPU_FUNCS(InterpolateLayer);
}