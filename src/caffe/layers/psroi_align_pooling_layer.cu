#include <cfloat>
#include "caffe/layers/psroi_align_pooling_layer.hpp"
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
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
__global__ void PSROIAlignPoolingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
    const int output_dim, // the channels of the output feature map
    const int group_size, // the size of the output feature map. pooled_height = group_size,pooled_width = group_size
    Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];

      Dtype roi_start_w = static_cast<Dtype>(bottom_rois[1]) * spatial_scale;
      Dtype roi_start_h = static_cast<Dtype>(bottom_rois[2]) * spatial_scale;
      Dtype roi_end_w = static_cast<Dtype>(bottom_rois[3]) * spatial_scale;
      Dtype roi_end_h = static_cast<Dtype>(bottom_rois[4]) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w,(Dtype)1);  // avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, (Dtype)1);

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      Dtype hstart = static_cast<Dtype>(ph) * bin_size_h + roi_start_h;
      Dtype wstart = static_cast<Dtype>(pw)* bin_size_w  + roi_start_w;
      Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h;
      Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w;

      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, (Dtype)0), (Dtype)height);
      hend = min(max(hend, (Dtype)0), (Dtype)height);
      wstart = min(max(wstart, (Dtype)0), (Dtype)width);
      wend = min(max(wend, (Dtype)0), (Dtype)width);

      bool is_empty = (hend <= hstart) || (wend <= wstart);

      int c = (ctop * group_size + ph) * group_size + pw;

      const Dtype *batch_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
      Dtype sum = 0,value;
      for (int i = 0;i < 2; ++i){
        for(int j = 0;j < 2; ++j){
          value = 0;
            Dtype h = hstart + i * bin_size_h;
            Dtype w = wstart + j * bin_size_w;
            bilinear_interpolate(batch_data,height,width,h,w,value);
            sum += value;
        }
      }
      top_data[index] = is_empty ? 0:sum / static_cast<Dtype>(4);
    }
  }
  template <typename Dtype>
  void PSROIAlignPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);

    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIAlignPoolingForward<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, spatial_scale_,
      channels_, height_, width_, pooled_height_,pooled_width_, 
      bottom_rois, output_dim_, group_size_,top_data);
    CUDA_POST_KERNEL_CHECK;
  }
template <typename Dtype>
__device__ Dtype get_feature_gradient(Dtype argmax_h, Dtype argmax_w, 
    const int h, const int w, const int height, const int width){
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
  __global__ void PSROIAlignBackwardFeature(
    const int nthreads,
    const Dtype* top_diff,
    const Dtype spatial_scale,
    const int channels,const int height, 
    const int width,const int group_size,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)

      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w = static_cast<Dtype>(bottom_rois[1]) * spatial_scale;
      Dtype roi_start_h = static_cast<Dtype>(bottom_rois[2]) * spatial_scale;
      Dtype roi_end_w = static_cast<Dtype>(bottom_rois[3]) * spatial_scale;
      Dtype roi_end_h = static_cast<Dtype>(bottom_rois[4]) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, (Dtype)1);  // avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, (Dtype)1);

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      Dtype hstart = static_cast<Dtype>(ph)* bin_size_h + roi_start_h;
      Dtype wstart = static_cast<Dtype>(pw)* bin_size_w + roi_start_w;
      Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h;
      Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w;
      
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, (Dtype)0), (Dtype)height);
      hend = min(max(hend, (Dtype)0), (Dtype)height);
      wstart = min(max(wstart, (Dtype)0), (Dtype)width);
      wend = min(max(wend, (Dtype)0), (Dtype)width);

      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c = (ctop * group_size + ph) * group_size + pw;
      int offset = (roi_batch_ind * channels + c) * height * width;

      for(int i = 0;i < 2; ++i){
        for(int j = 0;j < 2; ++j){

            Dtype index_h = hstart + i * bin_size_h;
            Dtype index_w = wstart + j * bin_size_w;

            Dtype weight_upper_left,weight_upper_right,weight_lower_left,weight_lower_right;

            weight_upper_left = get_feature_gradient(index_h,index_w,(int)index_h,(int)index_w,height,width);
            weight_upper_right = get_feature_gradient(index_h,index_w,(int)index_h,(int)index_w+1,height,width);
            weight_lower_left = get_feature_gradient(index_h,index_w,(int)index_h+1,(int)index_w,height,width);
            weight_lower_right = get_feature_gradient(index_h,index_w,(int)index_h+1,(int)index_w+1,height,width);

            const int offset_upper_left =  offset + ((int)index_h) * width + (int)index_w;
            const int offset_upper_right = offset + ((int)index_h) * width + (int)index_w + 1;
            const int offset_lower_left = offset + ((int)index_h + 1) * width +(int)index_w;
            const int offset_lower_right = offset + ((int)index_h + 1) * width +(int)index_w + 1;

            bottom_diff[offset_upper_left] += (is_empty) ? 0.: 0.25 * weight_upper_left * top_diff[index];
            bottom_diff[offset_upper_right] += (is_empty) ? 0.: 0.25 * weight_upper_right * top_diff[index];
            bottom_diff[offset_lower_left] += (is_empty) ? 0.: 0.25 * weight_lower_left * top_diff[index];
            bottom_diff[offset_lower_right] += (is_empty) ? 0.: 0.25 * weight_lower_right * top_diff[index];   
        }
      }
    }
  }  

template <typename Dtype>
void PSROIAlignPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
        return;
    }
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();

    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
    
    const int count = top[0]->count();
    
    // NOLINT_NEXT_LINE(whitespace/operators)

    PSROIAlignBackwardFeature<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, spatial_scale_,
      channels_, height_, width_,group_size_,pooled_height_, 
      pooled_width_, output_dim_, bottom_diff , bottom_rois);
    CUDA_POST_KERNEL_CHECK;
  
}
  
INSTANTIATE_LAYER_GPU_FUNCS(PSROIAlignPoolingLayer);
  
}  // namespace caffe