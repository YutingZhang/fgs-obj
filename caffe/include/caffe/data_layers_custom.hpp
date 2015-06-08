/*
 * data_layers_custom.hpp
 *
 *  Created on: Jun 11, 2014
 *      Author: zhangyuting
 */

#ifndef CAFFE_DATA_LAYERS_CUSTOM_HPP_
#define CAFFE_DATA_LAYERS_CUSTOM_HPP_

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/fake_leveldb.hpp"

#include <fstream>
#include <vector>
#include <list>

namespace caffe {

// This function is used to create a pthread that prefetches the window data.
struct FastWindowDataLayer_Aux {
    typedef std::vector<char> raw_image_t;
    typedef std::pair<std::string,raw_image_t> x_image_t;
};

template <typename Dtype>
void* FastWindowDataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class FastWindowDataLayer : public Layer<Dtype> {
    // The function used to perform prefetching.
    friend void* FastWindowDataLayerPrefetch<Dtype>(void* layer_pointer);

public:
    explicit FastWindowDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
    virtual ~FastWindowDataLayer();
    virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                       vector<Blob<Dtype>*>* top);

protected:
    virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             vector<Blob<Dtype>*>* top);
    virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             vector<Blob<Dtype>*>* top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
return; }
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
return; }

    pthread_t thread_;
    shared_ptr<Blob<Dtype> > prefetch_data_;
    shared_ptr<Blob<Dtype> > prefetch_label_;
    shared_ptr<Blob<Dtype> > prefetch_overlap_;
    Blob<Dtype> data_mean_;
    vector<std::pair<FastWindowDataLayer_Aux::x_image_t, vector<int> > >
image_database_;
    enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2,
        GX1, GY1, GX2, GY2, NUM };
    vector<vector<float> > fg_windows_;
    vector<vector<float> > amb_windows_;
    vector<vector<float> > bg_windows_;

    bool pair_pos_with_groundtruth_;
    bool include_ambiguous_pos_;
    bool output_overlap_;

    vector< FastWindowDataLayer_Aux::raw_image_t* > raw_image_ptr_;
    vector< vector<float> > cur_windows_;
    vector< bool > do_mirror_list_;
public:
    bool pair_pos_with_groundtruth() const { return
pair_pos_with_groundtruth_; }
    bool include_ambiguous_pos() const { return include_ambiguous_pos_; }
    bool output_overlap() const { return output_overlap_; }
};



}

#endif

