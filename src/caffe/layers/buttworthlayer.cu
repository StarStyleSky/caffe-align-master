#include "caffe/layers/butterworthlayer.hpp"
#include "caffe/util/math_functions.hpp"
#include <algorithm>
#include <utility>
namespace caffe{
	template<typename Dtype>
	__global__ void Butterworth_Forward(const int nthreads,
		const Dtype* bottom_data,const Dtype eps,
		const int channels,const int height,
		const int width,const Dtype exp,
		const Dtype cut_off,Dtype* top_data){
		CUDA_KERNEL_LOOP(index, nthreads) {

    		int n = index / width / height / channels;
    		Dtype maxval,minval;
    		int offset = channels* height * width;
    		const Dtype* batch_data = bottom_data + n * offset;

    		maxval = batch_data[0];
    		minval = batch_data[0];
    		for(int i = 1;i<offset;++i){
    			if (maxval < batch_data[i]){
    				maxval = batch_data[i];
    			}
    			if (minval > batch_data[i]){
    				minval = batch_data[i];
    			}
    		} 
    		top_data[index] = (bottom_data[index] - minval)/(maxval-minval + eps);
    		top_data[index] = Dtype(1.) - Dtype(1.)/(Dtype(1.)+pow(static_cast<double>   \
    		  (bottom_data[index]/cut_off),static_cast<double>(exp)));
    		maxval = top_data[0];
    		minval = top_data[0];
    		for(int i = 1;i<offset;++i){
    			if (maxval < top_data[i]){
    				maxval = top_data[i];
    			}
    			if (minval > top_data[i]){
    				minval = top_data[i];
    			}
    		}
    		top_data[index] = (top_data[index] - minval)/(maxval-minval + eps);
    	}
	}
	template<typename Dtype>
	void ButterworthLayer<Dtype>:: Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();		

		const int  count = bottom[0]->count();
		Dtype exponent = Dtype(2.)*Dtype(orders_);

		Butterworth_Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
		(count,bottom_data,eps_,channels_,height_,width_,exponent,cut_off_,top_data);
		CUDA_POST_KERNEL_CHECK;
	}
	template<typename Dtype>
	void ButterworthLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		 NOT_IMPLEMENTED; 		
	}
	INSTANTIATE_LAYER_GPU_FUNCS(ButterworthLayer);
}