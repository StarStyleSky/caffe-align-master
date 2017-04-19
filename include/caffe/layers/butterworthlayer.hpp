//---------------create by yexiguafuqihao-----------------------
#ifndef CAFFE_BUTTERWORTH_LAYER_HPP_
#define CAFFE_BUTTERWORTH_LAYER_HPP_
#include <vector>
#include <cfloat>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
	template <typename Dtype>
	class ButterworthLayer : public Layer<Dtype>{

	public:
		explicit ButterworthLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Butterworth"; }				
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MaxBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 1; }

	protected:

		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		

		int num_;
		int channels_;
		int height_;
		int width_;
		
		Dtype cut_off_;
		Dtype eps_;
		int orders_;

		Blob<Dtype> forward_buffer;
	};

}
#endif