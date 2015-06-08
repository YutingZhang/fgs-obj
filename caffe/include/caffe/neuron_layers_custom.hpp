/*
 * neuron_layers_custom.hpp
 *
 *  Created on: Jun 11, 2014
 *      Author: zhangyuting
 */

#ifndef CAFFE_NEURON_LAYERS_CUSTOM_HPP_
#define CAFFE_NEURON_LAYERS_CUSTOM_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class NaN2ZeroLayer : public NeuronLayer<Dtype> {
public:
    explicit NaN2ZeroLayer(const LayerParameter& param)
    : NeuronLayer<Dtype>(param) {}

protected:
    virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             vector<Blob<Dtype>*>* top);
    virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             vector<Blob<Dtype>*>* top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                               const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                               const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
};

}

#endif /* NEURON_LAYERS_CUSTOM_HPP_ */
