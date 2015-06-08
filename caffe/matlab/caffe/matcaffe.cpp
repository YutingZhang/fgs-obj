//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#define MAT_CAFFE_VERSION 20140912

#include <string>
#include <vector>
#include <list>
#include <map>

#include "mex.h"

#include "caffe/caffe.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

using namespace caffe;  // NOLINT(build/namespaces)

// The pointer to the internal caffe::Net instance
static shared_ptr<Net<float> > net_;
static int init_key = -2;

// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order
//   [width, height, channels, images]
// where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct
// format:
//   % convert from uint8 to single
//   im = single(im);
//   % reshape to a fixed size (e.g., 227x227)
//   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
//   % permute from RGB to BGR and subtract the data mean (already in BGR)
//   im = im(:,:,[3 2 1]) - data_mean;
//   % flip width and height to make width the fastest dimension
//   im = permute(im, [2 1 3]);
//
// If you have multiple images, cat them with cat(4, ...)
//
// The actual forward function. It takes in a cell array of 4-D arrays as
// input and outputs a cell array.

static void do_save_net( const std::string& filename ) {
	NetParameter net_param;
	net_->ToProto(&net_param);
	WriteProtoToBinaryFile(net_param, filename.c_str());
}

static void do_set_layer_weights(
        const int layer_ids1, const mxArray* const layer_weights ) {
    const int layer_ids = layer_ids1 - 1; // 1-base to 0-base
    if (layer_ids<0) {
        LOG(ERROR) << "Wrong layer id (less than 0)";
        mexErrMsgTxt("Wrong layer id (less than 0)");
    }
    
    const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
    const vector<string>& layer_names = net_->layer_names();
    
    // find the target layer blobs
    vector<shared_ptr<Blob<float> > >* layer_blobs_ptr = NULL;
    std::string layer_name;
    int cur_layer_ids = 0;
    {
        string prev_layer_name = "";
        for (unsigned int i = 0; i < layers.size(); ++i) {
          vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
          if (layer_blobs.size() == 0) {
            continue;
          }
          if ( cur_layer_ids == layer_ids ) {
              layer_blobs_ptr = &layer_blobs;
              layer_name = layer_names[i];
              break;
          }
          if (layer_names[i] != prev_layer_name) {
            prev_layer_name = layer_names[i];
            cur_layer_ids++;
          }
        }
    }
    
    if (!layer_blobs_ptr) {
        LOG(ERROR) << "Wrong layer id (too large)";
        mexErrMsgTxt("Wrong layer id (too large)");
    }
    
    vector<shared_ptr<Blob<float> > >& layer_blobs = *layer_blobs_ptr;
    
    // check the input
    {
        const size_t layer_weights_numel = mxGetNumberOfElements(layer_weights);
        if ( layer_weights_numel != layer_blobs.size() ) {
            LOG(ERROR) << "Wrong blob number";
            mexErrMsgTxt("Wrong blob number");
        }
    }
    
    for ( size_t i = 0; i<layer_blobs.size(); ++i ) {
        const mxArray* const cur_weight = mxGetCell( layer_weights, static_cast<mwIndex>(i) );
        if ( !mxIsSingle(cur_weight) ) {
            LOG(ERROR) << "Weights must be SINGLE";
            mexErrMsgTxt("Weights must be SINGLE");
        }
        const mwSize  cur_weight_ndim = mxGetNumberOfDimensions( cur_weight );
        const mwSize* cur_weight_dims = mxGetDimensions( cur_weight );
        
        mwSize idims[4] = {1, 1, 1, 1};
        if (cur_weight_ndim<=4) {
            for (mwSize k=0;k<cur_weight_ndim;++k) 
                idims[k] = cur_weight_dims[k];
        } else {
            LOG(ERROR) << "Invalid weights dims";
            mexErrMsgTxt("Invalid weights dims");
        };
        
        mwSize dims[4] = {layer_blobs[i]->width(), layer_blobs[i]->height(),
             layer_blobs[i]->channels(), layer_blobs[i]->num()};
            
        for ( int j=0; j<4;++j ) {
            if ( idims[j] != dims[j] ) {
                LOG(ERROR) << "Wrong weights dims for Blob " << i 
                    << " in Layer " << layer_name;
                LOG(ERROR) << "Internal dims: " 
                    << dims[0] << ", "
                    << dims[1] << ", "
                    << dims[2] << ", "
                    << dims[3];
                LOG(ERROR) << "Input dims:    " 
                    << idims[0] << ", "
                    << idims[1] << ", "
                    << idims[2] << ", "
                    << idims[3];
                LOG(ERROR) << "Input ndims :  " << cur_weight_ndim;
                mexErrMsgTxt("Wrong weights dims");
            }
        }
    }
    
    // copy the content
    for ( size_t i = 0; i<layer_blobs.size(); ++i ) {
        const mxArray* const cur_weight = mxGetCell( layer_weights, static_cast<mwIndex>(i) );
        
		const float* weights_ptr = reinterpret_cast<float*>(mxGetPr( cur_weight ));

        //  mexPrintf("layer: %s (%d) blob: %d  %d: (%d, %d, %d) %d\n",
        //  layer_names[i].c_str(), i, j, layer_blobs[j]->num(),
        //  layer_blobs[j]->height(), layer_blobs[j]->width(),
        //  layer_blobs[j]->channels(), layer_blobs[j]->count());

        switch (Caffe::mode()) {
        case Caffe::CPU:
          memcpy(layer_blobs[i]->mutable_cpu_data(), weights_ptr,
              sizeof(float) * layer_blobs[i]->count());
          break;
        case Caffe::GPU:
          CUDA_CHECK(cudaMemcpy(layer_blobs[i]->mutable_gpu_data(), weights_ptr,
              sizeof(float) * layer_blobs[i]->count(), cudaMemcpyHostToDevice));
          break;
        default:
          LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
        
    }

}

static mxArray* do_get_layer_names() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  std::list<std::string> layer_names_list;
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        layer_names_list.push_back( prev_layer_name );
        num_layers++;
      }
    }
  }
  
  mxArray* l = mxCreateCellMatrix(num_layers, 1);
  {
      std::list<std::string>::iterator iter = layer_names_list.begin();
      for ( int i = 0; i<num_layers; ++i ) {
          mxSetCell( l, i, mxCreateString( iter->c_str() ) );
          ++iter;
      }
  }
  
  return l;
  
}

static mxArray* do_get_response_ids( const vector<Blob<float>*>& target_blobs ) {
    const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();

    typedef map<const Blob<float>*, size_t> blob_id_map_t;
    blob_id_map_t bim;
    for ( size_t i=0; i<blobs.size(); ++i )
        bim[blobs[i].get()] = i + 1;  // to 1-base

    vector<size_t> target_ids( target_blobs.size() );
    for ( size_t i=0; i<target_blobs.size(); ++i ) {
        target_ids[i] = bim[target_blobs[i]];  // if cannot find, give 0
    }

    mxArray* mx_out = mxCreateDoubleMatrix( target_blobs.size(), 1, mxREAL );
    double* target_ids_double = mxGetPr(mx_out);
    for ( size_t i=0; i<target_blobs.size(); ++i ) {
        target_ids_double[i] = static_cast<double>(target_ids[i]);
    }

    return mx_out;
}

static mxArray* do_get_input_response_ids() {
	return do_get_response_ids( net_->input_blobs() );
}

static mxArray* do_get_output_response_ids() {
	return do_get_response_ids( net_->output_blobs() );
}

static mxArray* do_get_response_info() {
	const vector<string>& blob_names = net_->blob_names();
	const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
	size_t n = blob_names.size();

	mxArray* mx_out;
	{
		const mwSize dims[2] = {n, 1};
		const char* fnames[2] = {"name", "size"};
		mx_out = mxCreateStructArray(2, dims, 2, fnames);
	}

	for (unsigned int i = 0; i < n; ++i) {
		mxArray* mx_blob_name = mxCreateString(blob_names[i].c_str());
		mxArray* mx_blob_size = mxCreateDoubleMatrix(1,4,mxREAL);
		double* mx_blob_size_double = mxGetPr( mx_blob_size );
		mx_blob_size_double[0] = blobs[i]->width();
		mx_blob_size_double[1] = blobs[i]->height();
		mx_blob_size_double[2] = blobs[i]->channels();
		mx_blob_size_double[3] = blobs[i]->num();
    	mxSetField(mx_out, i, "name", mx_blob_name);
    	mxSetField(mx_out, i, "size",mx_blob_size);
	}
	return mx_out;
}

static void do_forward_no_output(const mxArray* const bottom) {
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(bottom)[0]),
      input_blobs.size());
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_gpu_data());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
}


static mxArray* do_response(const vector<Blob<float>*>& target_blobs) {
	size_t n = target_blobs.size();
	mxArray* mx_out = mxCreateCellMatrix(n, 1);
	for (unsigned int i = 0; i < n; ++i) {
	    // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {target_blobs[i]->width(), target_blobs[i]->height(),
            target_blobs[i]->channels(), target_blobs[i]->num()};
        mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        // OLD: output a vector instead of an array
 	    // mxArray* mx_blob =  mxCreateNumericMatrix(target_blobs[i]->count(), 1,
		//     mxSINGLE_CLASS, mxREAL);
		mxSetCell(mx_out, i, mx_blob);
		float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
		switch (Caffe::mode()) {
		case Caffe::CPU:
			memcpy(data_ptr, target_blobs[i]->cpu_data(),
					sizeof(float) * target_blobs[i]->count());
			break;
		case Caffe::GPU:
			cudaMemcpy(data_ptr, target_blobs[i]->gpu_data(),
					sizeof(float) * target_blobs[i]->count(),
					cudaMemcpyDeviceToHost);
			break;
		default:
			LOG(FATAL)<< "Unknown Caffe mode.";
		}  // switch (Caffe::mode())
	}

	return mx_out;
}

static mxArray* do_response(const mxArray* const blob_ids) {

	const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
	size_t blob_num = net_->blobs().size();
	size_t n = static_cast<size_t>(mxGetM(blob_ids) * mxGetN(blob_ids));
	double* ids_double = mxGetPr(blob_ids);
	vector<Blob<float>*> output_blobs(n);
	for ( size_t i=0; i<n; ++i ) {
		size_t ids_i  = static_cast<size_t>(ids_double[i]) - 1; // 1-base to 0-base
		if (ids_i>=blob_num) {
			LOG(ERROR) << "Wrong blob id";
			mexErrMsgTxt("Wrong blob id");
		}
		output_blobs[i] = blobs[ids_i].get();
	}

	return do_response( output_blobs );

} // do_response

static mxArray* do_backward(const mxArray* const top_diff) {
  vector<Blob<float>*>& output_blobs = net_->output_blobs();
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(top_diff)[0]),
      output_blobs.size());
  // First, copy the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_cpu_diff());
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_gpu_diff());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  // LOG(INFO) << "Start";
  net_->Backward();
  // LOG(INFO) << "End";
  mxArray* mx_out = mxCreateCellMatrix(input_blobs.size(), 1);
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {input_blobs[i]->width(), input_blobs[i]->height(),
      input_blobs[i]->channels(), input_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->cpu_diff(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->gpu_diff(), data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  
  return mx_out;
}

static mxArray* do_forward(const mxArray* const bottom) {
  do_forward_no_output( bottom );
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  mxArray* mx_out = do_response( output_blobs );
  return mx_out;
}

static mxArray* do_get_weights() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  // Step 1: count the number of layers
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"weights", "layer_names"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
        mx_layer_cells = mxCreateCellArray(2, dims);
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
        mxSetField(mx_layers, mx_layer_index, "layer_names",
            mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};
        mxArray* mx_weights = mxCreateNumericArray(4, dims, mxSINGLE_CLASS,
                                                   mxREAL);
        mxSetCell(mx_layer_cells, j, mx_weights);
        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

        //  mexPrintf("layer: %s (%d) blob: %d  %d: (%d, %d, %d) %d\n",
        //  layer_names[i].c_str(), i, j, layer_blobs[j]->num(),
        //  layer_blobs[j]->height(), layer_blobs[j]->width(),
        //  layer_blobs[j]->channels(), layer_blobs[j]->count());

        switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),
              weights_ptr);
          break;
        case Caffe::GPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),
              weights_ptr);
          break;
        default:
          LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
      }
    }
  }

  return mx_layers;
}

static void get_weights(MEX_ARGS) {
  plhs[0] = do_get_weights();
}

static void set_mode_cpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::CPU);
}

static void set_mode_gpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::GPU);
}

static void set_phase_train(MEX_ARGS) {
  Caffe::set_phase(Caffe::TRAIN);
}

static void set_phase_test(MEX_ARGS) {
  Caffe::set_phase(Caffe::TEST);
}

static void set_device(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

static void get_init_key(MEX_ARGS) {
  plhs[0] = mxCreateDoubleScalar(init_key);
}

static void init(MEX_ARGS) {
  if (nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  char* param_file = mxArrayToString(prhs[0]);
  char* model_file = mxArrayToString(prhs[1]);

  net_.reset(new Net<float>(string(param_file)));
  if ( !string(model_file).empty() ) // if empty filename then, randomly initialized, or not intialized
    net_->CopyTrainedLayersFrom(string(model_file));

  mxFree(param_file);
  mxFree(model_file);

  // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
  init_key = rand();
  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}

static void reset(MEX_ARGS) {
  if (net_) {
    net_.reset();
    init_key = -2;
    LOG(INFO) << "Network reset, call init before use it again";
  }
}

static void forward(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  if (nlhs==1)
	  plhs[0] = do_forward(prhs[0]);
  else if (nlhs==0)
	  do_forward_no_output(prhs[0]);
  else {
	  LOG(ERROR) << "Only given " << nlhs << " arguments";
	  mexErrMsgTxt("Wrong number of outputs");
  }
}

static void response(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  if (!mxIsDouble( prhs[0] )) {
	LOG(ERROR) << "Arguments should be double";
	mexErrMsgTxt("Arguments should be double");
  }
  if ( mxGetM( prhs[0] )>1. && mxGetN( prhs[0] ) ) {
		LOG(ERROR) << "The argument should be a vector";
		mexErrMsgTxt("The argument should be a vector");
  }
  plhs[0] = do_response(prhs[0]);
}

static void get_response_info(MEX_ARGS) {
	if (nrhs != 0) {
		LOG(ERROR) << "Only given " << nrhs << " arguments";
		mexErrMsgTxt("Wrong number of arguments");
	}
	plhs[0] = do_get_response_info();
}

static void get_output_response_ids(MEX_ARGS) {
	if (nrhs != 0) {
		LOG(ERROR) << "Only given " << nrhs << " arguments";
		mexErrMsgTxt("Wrong number of arguments");
	}
	plhs[0] = do_get_output_response_ids();
}

static void get_input_response_ids(MEX_ARGS) {
	if (nrhs != 0) {
		LOG(ERROR) << "Only given " << nrhs << " arguments";
		mexErrMsgTxt("Wrong number of arguments");
	}
	plhs[0] = do_get_input_response_ids();
}

static void backward(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  plhs[0] = do_backward(prhs[0]);
}

static void is_initialized(MEX_ARGS) {
  if (!net_) {
    plhs[0] = mxCreateDoubleScalar(0);
  } else {
    plhs[0] = mxCreateDoubleScalar(1);
  }
}

static void get_layer_names(MEX_ARGS) {
  if (nrhs != 0) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  plhs[0] = do_get_layer_names();
}

static void set_layer_weights(MEX_ARGS) {
  if (nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  
  if (!mxIsCell(prhs[1])) {
    LOG(ERROR) << "The second argument should be a cell array";
    mexErrMsgTxt("The second argument should be a cell array");
  }
  
  do_set_layer_weights( static_cast<int>( mxGetScalar(prhs[0]) ), prhs[1] );
}

static void save_net(MEX_ARGS) {
	if (nrhs != 1) {
		LOG(ERROR) << "Only given " << nrhs << " arguments";
		mexErrMsgTxt("Wrong number of arguments");
	}
	if ( !mxIsChar(prhs[0]) ) {
		LOG(ERROR) << "The argument should be string";
		mexErrMsgTxt("The argument should be string");
	}
	char* net_file_ = mxArrayToString(prhs[0]);
	std::string net_file(net_file_);
	mxFree(net_file_);
	do_save_net( net_file );
}

static void get_version(MEX_ARGS) {
	if (nrhs != 0) {
		LOG(ERROR) << "Only given " << nrhs << " arguments";
		mexErrMsgTxt("Wrong number of arguments");
	}
	
	plhs[0] = mxCreateDoubleScalar(MAT_CAFFE_VERSION);
}

static void read_mean(MEX_ARGS) {
    if (nrhs != 1) {
        mexErrMsgTxt("Usage: caffe('read_mean', 'path_to_binary_mean_file'");
        return;
    }
    const string& mean_file = mxArrayToString(prhs[0]);
    Blob<float> data_mean;
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    bool result = ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
    if (!result) {
        mexErrMsgTxt("Couldn't read the file");
        return;
    }
    data_mean.FromProto(blob_proto);
    mwSize dims[4] = {data_mean.width(), data_mean.height(),
                      data_mean.channels(), data_mean.num() };
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    caffe_copy(data_mean.count(), data_mean.cpu_data(), data_ptr);
    mexWarnMsgTxt("Remember that Caffe saves in [width, height, channels]"
                  " format and channels are also BGR!");
    plhs[0] = mx_blob;
}

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "save_net",           save_net },
  { "set_layer_weights",  set_layer_weights },
  { "get_layer_names",    get_layer_names },
  { "get_input_response_ids", get_input_response_ids },
  { "get_output_response_ids", get_output_response_ids },
  { "get_response_info",  get_response_info },
  { "get_response",       response        },
  { "response",           response        }, // same as get response
  { "get_version",        get_version     },

  { "forward",            forward         },
  { "backward",           backward        },
  { "init",               init            },
  { "is_initialized",     is_initialized  },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_phase_train",    set_phase_train },
  { "set_phase_test",     set_phase_test  },
  { "set_device",         set_device      },
  { "get_weights",        get_weights     },
  { "get_init_key",       get_init_key    },
  { "reset",              reset           },
  { "read_mean",          read_mean       },
  // The end.
  { "END",                NULL            },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  if (nrhs == 0) {
    LOG(ERROR) << "No API command given";
    mexErrMsgTxt("An API command is requires");
    return;
  }

  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      LOG(ERROR) << "Unknown command `" << cmd << "'";
      mexErrMsgTxt("API command not recognized");
    }
    mxFree(cmd);
  }
}
