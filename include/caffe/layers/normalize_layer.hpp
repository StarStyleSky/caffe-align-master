//---------------create by yexiguafuqihao-----------------------
#ifndef CAFFE_NORMALIZATION_LAYER_HPP_
#define CAFFE_NORMALIZATION_LAYER_HPP_
#include <vector>
#include <cfloat>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/filler.hpp"
namespace caffe{
	template <typename Dtype>
	class NormalizeLayer : public Layer<Dtype>{

	public:
		explicit NormalizeLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Normalize"; }				
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

		Dtype eps_;
		Blob<Dtype> buffer_;
		Blob<Dtype> buffer_spatial_;
		Blob<Dtype> buffer_channel_;
		Blob<Dtype> norm_;
	
		Blob<Dtype> sum_channel_multiplier_;
		Blob<Dtype> sum_spatial_multiplier_;
		bool channel_shared_;
		bool across_spatial_;
	};

}
#endif