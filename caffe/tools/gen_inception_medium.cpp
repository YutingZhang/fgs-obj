#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "fcntl.h"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

const char* traindb = 
  "/mnt/neocortex/scratch/zhangyuting/data/imagenet-fakeleveldb/train/";
const char* valdb = 
  "/mnt/neocortex/scratch/zhangyuting/data/imagenet-fakeleveldb/val/";
const char* meanfile =
  "/mnt/neocortex/scratch/zhangyuting/imagenet/imagenet-mean.binaryproto";


void AddLinearLayer(const string& name, const string& bottom,
                    int size, LayerParameter* linear) {
  linear->set_name(name);
  linear->set_type(LayerParameter::INNER_PRODUCT);
  linear->add_bottom(bottom);
  linear->add_top(name);
  linear->add_blobs_lr(1);
  linear->add_blobs_lr(2);
  linear->add_weight_decay(1);
  linear->add_weight_decay(0);
  InnerProductParameter inner_par;
  inner_par.set_num_output(size);
  FillerParameter fill_weights;
  fill_weights.set_type("gaussian");
  fill_weights.set_std(0.01);
  (*inner_par.mutable_weight_filler()) = fill_weights;
  FillerParameter fill_bias;
  fill_bias.set_type("constant");
  fill_bias.set_value(0);
  (*inner_par.mutable_bias_filler()) = fill_bias;
  (*linear->mutable_inner_product_param()) = inner_par;
}

void AddDropoutLayer(const string& name, const string& bottom,
                     float drop_ratio, LayerParameter* drop) {
  drop->set_name(name);
  drop->set_type(LayerParameter::DROPOUT);
  drop->add_bottom(bottom);
  drop->add_top(name);
  DropoutParameter drop_params;
  drop_params.set_dropout_ratio(drop_ratio);
  (*drop->mutable_dropout_param()) = drop_params;
}

void AddConvolutionLayer(const string& name, const string& bottom,
                         int depth, int kernel_size, int padding,
                         int stride, LayerParameter* conv) {
  conv->set_name(name);
  conv->set_type(LayerParameter::CONVOLUTION);
  conv->add_bottom(bottom);
  conv->add_top(name);
  conv->add_blobs_lr(1);
  conv->add_blobs_lr(2);
  conv->add_weight_decay(1);
  conv->add_weight_decay(0);
  ConvolutionParameter conv_params;
  conv_params.set_num_output(depth);
  conv_params.set_kernel_size(kernel_size);
  conv_params.set_stride(stride);
  conv_params.set_pad(padding);
  FillerParameter fill_weights;
  fill_weights.set_type("gaussian");
  fill_weights.set_std(0.01);
  (*conv_params.mutable_weight_filler()) = fill_weights;
  FillerParameter fill_bias;
  fill_bias.set_type("constant");
  fill_bias.set_value(0);
  (*conv_params.mutable_bias_filler()) = fill_bias;
  *(conv->mutable_convolution_param()) = conv_params;
}

void AddReluLayer(const string& name, const string& bottom,
                  LayerParameter* relu) {
  relu->set_name(name);
  relu->set_type(LayerParameter::RELU);
  relu->add_bottom(bottom);
  relu->add_top(bottom);
}

void AddPoolingLayer(const string& name, const string& bottom,
                     int kernel_size, int stride, int pad,
                     PoolingParameter::PoolMethod method,
                     LayerParameter* pool) {
  pool->set_name(name);
  pool->set_type(LayerParameter::POOLING);
  pool->add_bottom(bottom);
  pool->add_top(name);
  PoolingParameter pooling_pars;
  pooling_pars.set_pool(method);
  pooling_pars.set_pad(pad);
  pooling_pars.set_kernel_size(kernel_size);
  pooling_pars.set_stride(stride);
  (*pool->mutable_pooling_param()) = pooling_pars;
}

void AddMixedLayer(const string& name, const string& bottom,
                   int conv_1x1_depth,
                   int bottleneck_3x3_depth,
                   int conv_3x3_depth,
                   int bottleneck_5x5_depth,
                   int conv_5x5_depth,
                   int bottleneck_pooling_depth,
                   NetParameter* net) {

  // 1x1 Convolution.
  AddConvolutionLayer(name + "_1x1_conv", bottom,
                      conv_1x1_depth, 1, 0, 1,
                      net->add_layers());
  AddReluLayer(name + "_1x1_relu", name + "_1x1_conv",
               net->add_layers());
  
  // 3x3 Bottleneck.
  AddConvolutionLayer(name + "_3x3_bottleneck", bottom,
                      bottleneck_3x3_depth, 1, 0, 1,
                      net->add_layers());
  AddReluLayer(name + "_3x3_bottleneck_relu", name + "_3x3_bottleneck",
               net->add_layers());

  // 3x3 Convolution.
  AddConvolutionLayer(name + "_3x3_conv", name + "_3x3_bottleneck",
                      conv_3x3_depth, 3, 1, 1,
                      net->add_layers());
  AddReluLayer(name + "_3x3_relu", name + "_3x3_conv",
               net->add_layers());

  // 5x5 Bottleneck.
  AddConvolutionLayer(name + "_5x5_bottleneck", bottom,
                      bottleneck_5x5_depth, 1, 0, 1,
                      net->add_layers());
  AddReluLayer(name + "_5x5_bottleneck_relu", name + "_5x5_bottleneck",
               net->add_layers());

  // 5x5 Convolution.
  AddConvolutionLayer(name + "_5x5_conv", name + "_5x5_bottleneck",
                      conv_5x5_depth, 5, 2, 1,
                      net->add_layers());
  AddReluLayer(name + "_5x5_relu", name + "_5x5_conv",
               net->add_layers());

  // Pooling.
  AddPoolingLayer(name + "_pooling", bottom, 3, 1, 1,
                  PoolingParameter::AVE, net->add_layers());

  // Pooling Bottleneck.
  AddConvolutionLayer(name + "_pooling_bottleneck", name + "_pooling",
                      bottleneck_pooling_depth, 1, 0, 1,
                      net->add_layers());
  AddReluLayer(name + "_pooling_relu", name + "_pooling_bottleneck",
               net->add_layers());

  // Depth concatenation.
  LayerParameter depth_concat;
  depth_concat.set_name(name);
  depth_concat.add_top(name);
  depth_concat.set_type(LayerParameter::CONCAT);
  depth_concat.add_bottom(name + "_1x1_conv");
  depth_concat.add_bottom(name + "_3x3_conv");
  depth_concat.add_bottom(name + "_5x5_conv");
  depth_concat.add_bottom(name + "_pooling_bottleneck");
  ConcatParameter concat_param;
  concat_param.set_concat_dim(1);
  (*depth_concat.mutable_concat_param()) = concat_param;
  (*net->add_layers()) = depth_concat;
}

void AddSoftmaxLayer(const string& name, const string& bottom,
                     LayerParameter* softmax) {
  softmax->set_name(name);
  softmax->add_top(name);
  softmax->set_type(LayerParameter::SOFTMAX);
  softmax->add_bottom(bottom);
}

void AddSoftmaxLoss(const string& name, const string& bottom,
                    const string& label, LayerParameter* softmax) {
  softmax->set_name(name);
  softmax->add_top(name);
  softmax->set_type(LayerParameter::SOFTMAX_LOSS);
  softmax->add_bottom(bottom);
  softmax->add_bottom(label);
}


void AddBody(NetParameter* net) {
  AddConvolutionLayer("conv1", "data", 96, 11, 0, 4, net->add_layers());
  AddReluLayer("relu1", "conv1", net->add_layers());
  AddPoolingLayer("pool1", "conv1", 3, 2, 0, PoolingParameter::MAX,
                  net->add_layers());

  // Inception layers - 27 x 27.
  AddMixedLayer("mixed2a", "pool1", 64, 96, 128, 16, 32, 32, net);

  // Dimensionality reduction - 13 x 13.
  AddConvolutionLayer("conv2", "mixed2a", 256, 3, 0, 2, net->add_layers());

  // Inception layers - 13 x 13.
  AddMixedLayer("mixed3a", "conv2", 128, 128, 192, 32, 48, 48, net);

  // Dimensionality reduction - 7 x 7.
  AddConvolutionLayer("conv3", "mixed3a", 480, 3, 0, 2, net->add_layers());

  // Inception layers - 7 x 7.
  AddMixedLayer("mixed4a", "conv3", 256, 160, 320, 32, 128, 128, net);

  // Average Pooling
  AddPoolingLayer("pool2", "mixed4a", 6, 1, 0, PoolingParameter::AVE, net->add_layers());

  // Dropout.
  AddDropoutLayer("drop", "pool2", 0.4, net->add_layers());

  // Linear.
  AddLinearLayer("linear", "drop", 1000, net->add_layers());
}

void AddAccuracyLayer(const string& name, const string& prob,
                      const string& label, LayerParameter* acc) {
  acc->set_name(name);
  acc->set_type(LayerParameter::ACCURACY);
  acc->add_bottom(prob);
  acc->add_bottom(label);
  acc->add_top(name);
}

void AddDataLayer(const string& name, const string& label,
                  const string& dbname, const string& meanfile,
                  int batchsize, int cropsize,
                  bool mirror, LayerParameter* data) {
  data->set_name(name);
  data->set_type(LayerParameter::IMAGEDB_DATA);
  data->add_top(name);
  data->add_top(label);
  ImagedbDataParameter imdb;
  imdb.set_source(dbname);
  imdb.set_mean_file(meanfile);
  imdb.set_batch_size(batchsize);
  imdb.set_crop_size(cropsize);
  imdb.set_mirror(mirror);
  (*data->mutable_imagedb_data_param()) = imdb;
}

// Generate inception network specifications.
int main(int argc, char** argv) {
  string prefix(argv[1]);

  // Deployment network spec.
  string deploy_fname = prefix + "_deploy.prototxt";
  NetParameter deploy;
  deploy.set_name("Inception");
  deploy.add_input("data");
  deploy.add_input_dim(10);
  deploy.add_input_dim(3);
  deploy.add_input_dim(227);
  deploy.add_input_dim(227);
  AddBody(&deploy);
  AddSoftmaxLayer("prob", "linear", deploy.add_layers());
  WriteProtoToTextFile(deploy, deploy_fname);

  // Training network spec.
  string train_fname = prefix + "_train.prototxt";
  NetParameter train;
  train.set_name("Inception");
  AddDataLayer("data", "label", traindb, meanfile, 64, 227, true,
               train.add_layers());
  AddBody(&train);
  AddSoftmaxLoss("prob", "linear", "label", train.add_layers());
  WriteProtoToTextFile(train, train_fname);

  // Validation network spec.
  string val_fname = prefix + "_val.prototxt";
  NetParameter val;
  val.set_name("Inception");
  AddDataLayer("data", "label", valdb, meanfile, 50, 227, false,
               val.add_layers());
  AddBody(&val);
  AddSoftmaxLayer("prob", "linear", val.add_layers());
  AddAccuracyLayer("accuracy", "prob", "label", val.add_layers());
  WriteProtoToTextFile(val, val_fname);
}

