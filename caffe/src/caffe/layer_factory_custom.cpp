/*
 * layer_factory_custom.cpp
 *
 *  Created on: Jun 11, 2014
 *      Author: zhangyuting
 */

#include <string>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;

namespace caffe {


// The layer factory function
template <typename Dtype>
Layer<Dtype>* GetLayerCustom(const LayerParameter& param) {
  const string& name = param.name();
  const LayerParameter_LayerType& type = param.type();
  switch (type) {
  case LayerParameter_LayerType_NAN2ZERO:
	return new NaN2ZeroLayer<Dtype>(param);
  case LayerParameter_LayerType_FAST_WINDOW_DATA:
	return new FastWindowDataLayer<Dtype>(param);
  }
  return (Layer<Dtype>*)(NULL);
}


template Layer<float>*  GetLayerCustom(const LayerParameter& param);
template Layer<double>* GetLayerCustom(const LayerParameter& param);


}
