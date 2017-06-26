#include "caffe/layers/butterworthlayer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe{
	template <typename Dtype>
	void ButterworthLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		
		CHECK(this->layer_param().butter_worth_param().cutoff() > 0 && this->layer_param().butter_worth_param().orders() >= 0)
			<< "the parameter of butterworth layer must be greater than or equal to 0.";
		cut_off_ = this->layer_param().butter_worth_param().cutoff();
		orders_ = this->layer_param().butter_worth_param().orders();
		eps_ = this->layer_param().butter_worth_param().eps();
	}
	template <typename Dtype>
	void ButterworthLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom.size(), Dtype(1)) << "bottom blob size must be 1.";

		num_ = bottom[0]->num();
		channels_ = bottom[0]->channels();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		
		top[0]->ReshapeLike(*bottom[0]);
		forward_buffer.ReshapeLike(*bottom[0]);
	}
	template<typename Dtype>
	void ButterworthLayer<Dtype>:: Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){

		const Dtype *bottom_data = bottom[0]->cpu_data();
		Dtype *top_data = top[0]->mutable_cpu_data();
		Dtype *buffer = forward_buffer.mutable_cpu_data();

		const int len = bottom[0]->count();
		caffe_set(len, Dtype(-FLT_MAX), top_data);	
		caffe_set(len, Dtype(-FLT_MAX), buffer);

		int n,idx;
		Dtype maxval,minval;
		Dtype exponent = Dtype(2.)*Dtype(orders_);
		Dtype scale = Dtype(1.)/cut_off_;
		const int count = bottom[0]->count(1);
		for (n = 0;n<num_;++n){
			const Dtype *batch_data = bottom_data+bottom[0]->offset(n);
			//find the maximal and minimal value to normalize the feature map 
			maxval = batch_data[0];
			minval = batch_data[0];
			for(idx = 1;idx<count;++idx){
				if(maxval<batch_data[idx]){
					maxval = batch_data[idx];
				}
				if(minval>batch_data[idx]){
					minval = batch_data[idx];
				}
			}
			for(idx = 0;idx<count;++idx)
				buffer[idx] = (batch_data[idx]-minval)/(maxval-minval + eps_);

			caffe_scal(count,scale,buffer);
			caffe_powx(count,forward_buffer.cpu_data(),exponent,buffer);
			caffe_add_scalar(count,Dtype(1),buffer);
			
			for(idx = 0;idx<count;++idx)
				top_data[idx] = Dtype(1.) - Dtype(1.)/forward_buffer.cpu_data()[idx];
			maxval = top_data[0];
			minval = top_data[0];
			for(idx = 1;idx < count;++idx){
				if(maxval<top_data[idx]){
					maxval = top_data[idx];
				}
				if(minval>top_data[idx]){
					minval = top_data[idx];
				}
			}
			for(idx = 0;idx < count; ++idx)
				top_data[idx] = (top_data[idx]-minval)/(maxval-minval + eps_);
			top_data += top[0]->offset(n);
		}

	}
	template<typename Dtype>
	void ButterworthLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		 NOT_IMPLEMENTED; 
		
	}
	INSTANTIATE_CLASS(ButterworthLayer);
	REGISTER_LAYER_CLASS(Butterworth);
}