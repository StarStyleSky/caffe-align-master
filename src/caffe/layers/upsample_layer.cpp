#include <cfloat>
#include "caffe/layers/upsample_layer.hpp"
using std::max;
using std::min;
using std::floor;
using std::ceil;
namespace caffe{
template <typename Dtype>
void UpsamplePoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
	UpsamplePoolingParameter upsample_pooling_param = this->layer_param_.upsample_pooling_param();
  	CHECK_GT(upsample_pooling_param.pooled_height(), 0)
      	<< "pooled_h must be > 0";
  	CHECK_GT(upsample_pooling_param.pooled_width(), 0)
      	<< "pooled_w must be > 0";
  	CHECK_EQ(upsample_pooling_param.has_pooled_height() == upsample_pooling_param.has_spatial_scale()
  		,upsample_pooling_param.has_spatial_scale() == upsample_pooling_param.has_pooled_width())
      	<<"pooled_height and pooled_width as well as spatial scale should be determined simutaneously.";
 
  	pooled_height_ = upsample_pooling_param.pooled_height();
  	pooled_width_ = upsample_pooling_param.pooled_width();
  	spatial_scale_ = upsample_pooling_param.spatial_scale();
}	
template <typename Dtype>
void UpsamplePoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
	const vector<Blob<Dtype>*>& top){
	channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
	width_ = bottom[0]->width();

  CHECK_EQ(height_ == static_cast<int>(spatial_scale_ * pooled_height_),
  	width_ == static_cast<int>(spatial_scale_ * pooled_width_))
  	<<"pooled size must be the same as the original input size";
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
}
template<typename Dtype>
void UpsamplePoolingLayer<Dtype>::bilinear_interpolate(const Dtype* bottom_data, const int height, 
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
void UpsamplePoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();

  int bottom_num = bottom[0]->num();

  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    
  for(int n = 0;n<bottom_num;++n){
    for (int c = 0; c < channels_; ++c) {  
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const Dtype* batch_data = bottom_data + bottom[0]->offset(n,c);
            Dtype value = 0;
            // mapping dest cordinate to src 
            // Dtype index_h = (ph + 0.5) * height_/pooled_height_ - 0.5;
            // Dtype index_w = (pw + 0.5) * width_/pooled_width_ - 0.5;
            Dtype index_h = static_cast<Dtype>(ph/spatial_scale_);
            Dtype index_w = static_cast<Dtype>(pw/spatial_scale_);
            bilinear_interpolate(batch_data, height_, width_, index_h, index_w, value); 
            // int index = ((n * channels_+ c) * pooled_height_ + ph) * pooled_width_ + pw;
            int index = top[0]->offset(n,c,ph,pw);
            top_data[index] = value;
          }
        }
      }
    }
  }
template<typename Dtype>
Dtype UpsamplePoolingLayer<Dtype>::get_feature_gradient(Dtype argmax_h, Dtype argmax_w, const int h,
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
void UpsamplePoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();
  caffe_set(count, Dtype(0.), bottom_diff);

  int top_num = top[0]->num();
  if(propagate_down[0]){
    for(int n = 0;n < top_num;++n){
      for(int c = 0;c < channels_; ++c){
        for(int ph = 0;ph < pooled_height_;++ph){
          for(int pw = 0;pw < pooled_width_ ;++ pw){
            //begining of (n,c) in bottom diff feature map
            Dtype *bottom_diff_data = bottom_diff + bottom[0]->offset(n,c);
            Dtype weight_upper_left,weight_upper_right,weight_lower_left,weight_lower_right;
            // mapping des cordinate into src
            // Dtype index_h = (ph + 0.5) * height_/pooled_height_-0.5;
            // Dtype index_w = (pw + 0.5) * width_/pooled_width_-0.5;
            Dtype index_h = static_cast<Dtype>(ph/spatial_scale_);
            Dtype index_w = static_cast<Dtype>(pw/spatial_scale_);

            weight_upper_left = get_feature_gradient(index_h,index_w,(int)index_h,(int)index_w,height_,width_);
            weight_upper_right = get_feature_gradient(index_h,index_w,(int)index_h,(int)index_w+1,height_,width_);
            weight_lower_left = get_feature_gradient(index_h,index_w,(int)index_h+1,(int)index_w,height_,width_);
            weight_lower_right = get_feature_gradient(index_h,index_w,(int)index_h+1,(int)index_w+1,height_,width_);

            // int index = ( (n * channels_ + c) * pooled_height_ + ph) * pooled_width_ + pw;
            int index = top[0]->offset(n,c,ph,pw);
            bottom_diff_data[((int)index_h)*width_+(int)index_w] += weight_upper_left * top_diff[index];
            bottom_diff_data[((int)index_h)*width_ + (int)index_w + 1] += weight_upper_right * top_diff[index];
            bottom_diff_data[((int)index_h + 1) * width_ + (int)index_w] += weight_lower_left * top_diff[index];
            bottom_diff_data[((int)index_h + 1) * width_ + (int)index_w + 1] += weight_lower_right * top_diff[index];
          }
        }
      }
    }
  }
}
#ifdef CPU_ONLY
STUB_GPU(UpsamplePoolingLayer);
#endif

INSTANTIATE_CLASS(UpsamplePoolingLayer);
REGISTER_LAYER_CLASS(UpsamplePooling);
}