#include <cfloat>
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/psroi_align_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void PSROIAlignPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    PSROIAlignPoolingParameter psroi_align_pooling_param =
    this->layer_param_.psroi_align_pooling_param();
    spatial_scale_ = psroi_align_pooling_param.spatial_scale();
    LOG(INFO) << "Spatial scale: " << spatial_scale_;

    CHECK_GT(psroi_align_pooling_param.output_dim(), 0)
    << "output_dim must be > 0";
    CHECK_GT(psroi_align_pooling_param.group_size(), 0)
    << "group_size must be > 0";

    output_dim_ = psroi_align_pooling_param.output_dim();
    group_size_ = psroi_align_pooling_param.group_size();
    pooled_height_ = group_size_;
    pooled_width_ = group_size_;
  }

  template <typename Dtype>
  void PSROIAlignPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    channels_ = bottom[0]->channels();
    CHECK_EQ(channels_, output_dim_*group_size_*group_size_)
    << "input channel number does not match layer parameters";
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();

    top[0]->Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
  }
  template<typename Dtype>
  void PSROIAlignPoolingLayer<Dtype>::bilinear_interpolate(const Dtype* bottom_data, const int height, 
    const int width, Dtype h, Dtype w, Dtype & value){
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
  void PSROIAlignPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // NOT_IMPLEMENTED;
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_rois = bottom[1]->cpu_data();
    // Number of ROIs
    int num_rois = bottom[1]->num();
    int batch_size = bottom[0]->num();
    int top_count = top[0]->count();
    Dtype* top_data = top[0]->mutable_cpu_data();
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
    for (int n = 0; n < num_rois; ++n) {
      int roi_batch_ind = bottom_rois[0];

      Dtype roi_start_w = static_cast<Dtype>(bottom_rois[1] * spatial_scale_);
      Dtype roi_start_h = static_cast<Dtype>(bottom_rois[2] * spatial_scale_);
      Dtype roi_end_w = static_cast<Dtype>(bottom_rois[3] * spatial_scale_);
      Dtype roi_end_h = static_cast<Dtype>(bottom_rois[4] * spatial_scale_);

      CHECK_GE(roi_batch_ind, 0);
      CHECK_LT(roi_batch_ind, batch_size);

      // force small ROI into 1x1
      Dtype roi_height = max(roi_end_h - roi_start_h , (Dtype)1.);
      Dtype roi_width = max(roi_end_w - roi_start_w , (Dtype)1.);

      const Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_);
      const Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width_);
      // c is the channel index in the top feature map
      for (int c = 0; c < output_dim_; ++c) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              // Compute pooling region for this output unit:
              //  start (included) = floor(ph * roi_height / pooled_height_)
              //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
              int channels = (c * group_size_ + ph) * group_size_ + pw;
              const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind, channels); 

              Dtype hstart = static_cast<Dtype>(ph* bin_size_h);
              Dtype wstart = static_cast<Dtype>(pw* bin_size_w);
              Dtype hend = static_cast<Dtype>((ph + 1)* bin_size_h);
              Dtype wend = static_cast<Dtype>((pw + 1)* bin_size_w);

              hstart = min(max(hstart + roi_start_h, (Dtype)0), static_cast<Dtype>(height_));
              hend = min(max(hend + roi_start_h, (Dtype)0), static_cast<Dtype>(height_));
              wstart = min(max(wstart + roi_start_w, (Dtype)0), static_cast<Dtype>(width_));
              wend = min(max(wend + roi_start_w, (Dtype)0), static_cast<Dtype>(width_));

              bool is_empty = (hend <= hstart) || (wend <= wstart);

              const int pool_index = ph * pooled_width_ + pw;
              
              Dtype sum = 0,value = 0;
              for(int i = 0;i<2;++i){
                for(int j = 0;j<2;++j){
                  Dtype h = hstart + i * bin_size_h;
                  Dtype w = wstart + j * bin_size_w;
                  bilinear_interpolate(batch_data,height_,width_,h,w,value);
                  sum += value;
                }
              }
              top_data[pool_index] = is_empty ? 0 : sum / 4;
            } 
          }
        // Increment all data pointers by one channel
        top_data += top[0]->offset(0, 1);
        } 
        // Increment ROI data pointer
        bottom_rois += bottom[1]->offset(1);
      }
  }

  template <typename Dtype>
  void PSROIAlignPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
#ifdef CPU_ONLY
  STUB_GPU(PSROIAlignPoolingLayer);
#endif

  INSTANTIATE_CLASS(PSROIAlignPoolingLayer);
  REGISTER_LAYER_CLASS(PSROIAlignPooling);

}  // namespace caffe
