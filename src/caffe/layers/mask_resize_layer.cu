// --------------------------------------------------------
// Multitask Network Cascade
// Written by Haozhi Qi
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include "caffe/layers/mask_resize_layer.hpp"

namespace caffe {
  
template <typename Dtype>
__device__ Dtype bilinear_interpolate(const Dtype* bottom_data, const int input_height, const int input_width, Dtype inverse_y, Dtype inverse_x) {
  
  // deal with cases that inverse elements are out of feature map boundary
  if (inverse_y < -0.5 || inverse_y > input_height - 0.5 || inverse_x < -0.5 || inverse_x > input_width - 0.5) {
    return 0.0;
  }
  
  if (inverse_y <= 0) inverse_y = 0;
  if (inverse_x <= 0) inverse_x = 0;
  
  int h_low = (int) inverse_y;
  int w_low = (int) inverse_x;
  int h_high;
  int w_high;
  
  if (h_low >= input_height - 1) {
    h_high = h_low = input_height - 1;
    inverse_y = (Dtype) h_low;
  } else {
    h_high = h_low + 1;
  }
  
  if (w_low >= input_width - 1) {
    w_high = w_low = input_width - 1;
    inverse_x = (Dtype) w_low;
  } else {
    w_high = w_low + 1;
  }
  
  Dtype lh = inverse_y - h_low;
  Dtype lw = inverse_x - w_low;
  Dtype hh = 1 - lh, hw = 1 - lw;
  // do bilinear interpolation
  Dtype v1 = bottom_data[h_low * input_width + w_low];
  Dtype v2 = bottom_data[h_low * input_width + w_high];
  Dtype v3 = bottom_data[h_high * input_width + w_low];
  Dtype v4 = bottom_data[h_high * input_width + w_high];
  Dtype w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename Dtype>
__global__ void MaskResizeForward(const int nthreads, const Dtype* bottom_data, const int output_width, const int output_height, const int output_channels, const int input_width, const int input_height, const int input_channels, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) is an element in output mask
    int w = index % output_width;
    int h = (index / output_width) % output_height;
    int c = (index / output_width / output_height) % output_channels;
    int n = index / output_width / output_height / output_channels;
    Dtype ratio_h = static_cast<Dtype>(input_height) / static_cast<Dtype>(output_height);
    Dtype ratio_w = static_cast<Dtype>(input_width) / static_cast<Dtype>(output_width);

 //   Dtype inverse_x = w * ratio_w;
 //  Dtype inverse_y = h * ratio_h;
    Dtype inverse_x = (w + 0.5) * ratio_w - 0.5;
    Dtype inverse_y = (h + 0.5) * ratio_h - 0.5;

    const Dtype* offset_bottom_data = bottom_data + (n * input_channels + c) * input_height * input_width; 
    top_data[index] = bilinear_interpolate(offset_bottom_data, input_height, input_width, inverse_y, inverse_x);
  }
} 
template <typename Dtype>
void MaskResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  MaskResizeForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, bottom_data, output_width_, output_height_, output_channels_, 
  input_width_, input_height_, input_channels_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__device__ Dtype getGradientWeight(Dtype argmax_h, Dtype argmax_w, const int h, const int w, const int height, const int width){
  if (argmax_h < -0.5 || argmax_h >(height - 0.5) || argmax_w < -0.5 || argmax_w >(width - 0.5)){
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
__global__ void MaskResizeBackward(const int nthreads, const Dtype* top_diff,
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

    weight_upper_left = getGradientWeight(index_h,index_w,(int)index_h,(int)index_w,height,width);
    weight_upper_right = getGradientWeight(index_h,index_w,(int)index_h,(int)index_w+1,height,width);
    weight_lower_left = getGradientWeight(index_h,index_w,(int)index_h+1,(int)index_w,height,width);
    weight_lower_right = getGradientWeight(index_h,index_w,(int)index_h+1,(int)index_w+1,height,width);

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
void MaskResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    int count = top[0]->count();
    caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
    if(propagate_down[0]){
      MaskResizeBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
      (count, top_diff, input_channels_, input_height_, 
      input_width_, output_height_,output_width_, bottom_diff);
    }
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(MaskResizeLayer);

}