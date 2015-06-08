/*
 * misc_data.hpp
 *
 *  Created on: Oct 25, 2014
 *      Author: zhangyuting
 */

#ifndef INCLUDE_CAFFE_UTIL_ZETA_MISC_DATA_HPP_
#define INCLUDE_CAFFE_UTIL_ZETA_MISC_DATA_HPP_

namespace zeta {

struct raw_data_block {
	const char* raw_data;
	size_t length;
public:
	bool operator == ( const raw_data_block& r_ ) const {
		return (raw_data==r_.raw_data && length == r_.length);
	}
	bool operator < ( const raw_data_block& r_ ) const {
		return (raw_data<r_.raw_data) || (raw_data==r_.raw_data && length < r_.length);
	}
	bool operator <= ( const raw_data_block& r_ ) const {
		return (raw_data<r_.raw_data) || (raw_data==r_.raw_data && length <= r_.length);
	}
	bool operator > ( const raw_data_block& r_ ) const {
		return (raw_data>r_.raw_data) || (raw_data==r_.raw_data && length > r_.length);
	}
	bool operator >= ( const raw_data_block& r_ ) const {
		return (raw_data>r_.raw_data) || (raw_data==r_.raw_data && length >= r_.length);
	}
};

typedef std::vector< char > vec_block_data;

}

#endif /* INCLUDE_CAFFE_UTIL_ZETA_MISC_DATA_HPP_ */
