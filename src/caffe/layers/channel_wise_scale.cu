#include <cfloat>
#include <vector>

#include "caffe/layers/channel_wise_scale.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe{
template <typename Dtype>
__global__ void ChannelWiseScaleForward(const int nthreads,const Dtype *bottom_data,
		const Dtype *scale_data,const int channels,const int height,const int width,
		Dtype *top_data) {
	CUDA_KERNEL_LOOP(index, nthreads){
		// (n, c, ph, pw) is an element in the pooled output
    	int c = (index / height / width) % channels;
    	int n = index / height / width / channels;

    	Dtype scale = *(scale_data + n * channels + c);  // get the scaler
    	top_data[index] = scale * bottom_data[index];
	}
}
template <typename Dtype>
void ChannelWiseScaleLayer<Dtype> :: Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* scale_factors = bottom[1]->gpu_data();
  	Dtype* top_data = top[0]->mutable_gpu_data();
  	int count = bottom[0]->count();
  	ChannelWiseScaleForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
  	(count,bottom_data,scale_factors,channels_,height_,width_,top_data);
  	CUDA_POST_KERNEL_CHECK;
}
template<typename Dtype>
__global__ void FeatureBackward(const int nthreads,const Dtype *top_diff,const Dtype *scale_data,
	const int channels,const int height,const int width,Dtype *bottom_data_diff){
	CUDA_KERNEL_LOOP(index, nthreads){
		int c = (index / height / width) % channels;
    	int n = index / height / width / channels;
    	Dtype scale = *(scale_data + n *channels +  c);
    	bottom_data_diff[index] = scale * top_diff[index];
	}
}
template<typename Dtype>
__global__ void ScaleBackward(const int nthreads,const Dtype *top_diff,const Dtype *bottom_data,
	const int channels,const int height,const int width,Dtype *bottom_scale_diff){
	CUDA_KERNEL_LOOP(index, nthreads){
		//(n,c,ph,pw) in top_diff and bottom_data feature map
    	int c = (index / height / width) % channels;
    	int n = index / height / width / channels;

    	int offset = n * channels + c;
    	bottom_scale_diff[offset] += top_diff[index]*bottom_data[index];
	}
}
template<typename Dtype>
void ChannelWiseScaleLayer<Dtype> :: Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* scale_factors = bottom[1]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype *bottom_data_diff = bottom[0]->mutable_gpu_diff();
	Dtype *bottom_scale_diff = bottom[1]->mutable_gpu_diff();

	caffe_gpu_set(bottom[0]->count(),Dtype(0),bottom_data_diff);
	caffe_gpu_set(bottom[1]->count(),Dtype(0),bottom_scale_diff);
	
	int count = bottom[0]->count();
	if(propagate_down[0]){

		FeatureBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
		(count,top_diff,scale_factors,channels_,height_,width_,bottom_data_diff);
	}
	if(propagate_down[1]){

		ScaleBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
		(count,top_diff,bottom_data,channels_,height_,width_,bottom_scale_diff);
	}
	CUDA_POST_KERNEL_CHECK;
}
INSTANTIATE_LAYER_GPU_FUNCS(ChannelWiseScaleLayer);
}