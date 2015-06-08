/*
 * math_functions_custom.hpp
 *
 *  Created on: Jun 10, 2014
 *      Author: zhangyuting
 */

#ifndef CAFFE_MATH_FUNCTIONS_CUSTOM_HPP_
#define CAFFE_MATH_FUNCTIONS_CUSTOM_HPP_

#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype caffe_gpu_asum(const int n, const Dtype* x) {
	Dtype y;
	caffe_gpu_asum(n, x, &y);
	return y;
}

}


#endif /* MATH_FUNCTIONS_CUSTOM_HPP_ */
