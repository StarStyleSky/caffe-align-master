#include <cfloat>
#include "caffe/fast_rcnn_layers.hpp"
using std::max;
using std::min;
using std::floor;
using std::ceil;
namespace caffe {
template <typename Dtype>
void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIAlignParameter roi_align_param = this->layer_param_.roi_align_param();
  CHECK_GT(roi_align_param.pooled_height(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_align_param.pooled_width(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_align_param.pooled_height();
  pooled_width_ = roi_align_param.pooled_width();
  spatial_scale_ = roi_align_param.spatial_scale();
}
template<typename Dtype>
void ROIAlignLayer<Dtype>::bilinear_interpolate(const Dtype* bottom_data, const int height, 
  const int width, Dtype h, Dtype w, Dtype & maxval, Dtype & maxidx_h, Dtype & maxidx_w){
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
  
  Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  if (val > maxval) {
    maxval = val;
    maxidx_h = h;
    maxidx_w = w;
  }
}
template <typename Dtype>
void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_);
  max_idx_w_.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_);
  max_idx_h_.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_);
  buffer_.Reshape(bottom[1]->num() * 5, channels_, pooled_height_, pooled_width_);
}
template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  	const Dtype* bottom_data = bottom[0]->cpu_data();
  	const Dtype* bottom_rois = bottom[1]->cpu_data();
  	// Number of ROIs
  	int num_rois = bottom[1]->num();
  	int batch_size = bottom[0]->num();
  	int top_count = top[0]->count();
  	Dtype* top_data = top[0]->mutable_cpu_data();
  	caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  	Dtype *max_idx_x = max_idx_h_.mutable_cpu_data();
  	Dtype *max_idx_y = max_idx_w_.mutable_cpu_data();
  	caffe_set(top_count, Dtype(-1), max_idx_x);
  	caffe_set(top_count, Dtype(-1),max_idx_y);

  	// For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  	for (int n = 0; n < num_rois; ++n) {
    	int roi_batch_ind = bottom_rois[0];

    	Dtype roi_start_w = static_cast<Dtype>(bottom_rois[1] * spatial_scale_);
    	Dtype roi_start_h = static_cast<Dtype>(bottom_rois[2] * spatial_scale_);
    	Dtype roi_end_w = static_cast<Dtype>(bottom_rois[3] * spatial_scale_);
    	Dtype roi_end_h = static_cast<Dtype>(bottom_rois[4] * spatial_scale_);

    	CHECK_GE(roi_batch_ind, 0);
    	CHECK_LT(roi_batch_ind, batch_size);

    	Dtype roi_height = max(roi_end_h - roi_start_h , (Dtype)1.);
    	Dtype roi_width = max(roi_end_w - roi_start_w , (Dtype)1.);
    	const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    	const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);
    	for (int c = 0; c < channels_; ++c) {
      		for (int ph = 0; ph < pooled_height_; ++ph) {
        		for (int pw = 0; pw < pooled_width_; ++pw) {
          		// Compute pooling region for this output unit:
          		//  start (included) = floor(ph * roi_height / pooled_height_)
          		//  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          		const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind,c);

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
          		if (is_empty) {
            		top_data[pool_index] = 0;
            		max_idx_x[pool_index] = -1;
          			max_idx_y[pool_index] = -1;
          		}
          		Dtype maxval = -1;
          		Dtype maxidx_h = -1;
          		Dtype maxidx_w = -1;
          		// Find the maximum value in the four corners of ROI bin and record its cordinate
          		for(int i = 0;i < 2;++i){
          			for(int j = 0;j < 2;++j){
          				Dtype h = hstart + i*bin_size_h;
          				Dtype w = wstart + j*bin_size_w;
          				bilinear_interpolate(batch_data,height_,width_,h,w,maxval,maxidx_h,maxidx_w);
          			}
          		}
          		if (maxidx_h == -1 && maxidx_w == -1) maxval = 0;
              	top_data[pool_index] = is_empty ? 0 : maxval;
          		max_idx_x[pool_index] = is_empty ? -1 : maxidx_h;
          		max_idx_y[pool_index] = is_empty ? -1 : maxidx_w;
        	  } 
      		}
     		// Increment all data pointers by one channel
     		top_data += top[0]->offset(0, 1);
     		max_idx_x += max_idx_h_.offset(0,1);
     		max_idx_y += max_idx_w_.offset(0,1);
    		}	
    		// Increment ROI data pointer
    		bottom_rois += bottom[1]->offset(1);
  		}
	}
template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}
#ifdef CPU_ONLY
STUB_GPU(ROIAlignLayer);
#endif

INSTANTIATE_CLASS(ROIAlignLayer);
REGISTER_LAYER_CLASS(ROIAlign);

}  // namespace caffe
