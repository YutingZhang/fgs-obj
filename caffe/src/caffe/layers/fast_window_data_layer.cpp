// Copyright 2014 Yuting Zhang
//
// Based on :
// Copyright 2013 Ross Girshick
//
// Based on data_layer.cpp by Yangqing Jia.

#include <stdint.h>
#include <pthread.h>

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <fstream>  // NOLINT(readability/streams)
#include <utility>
#include <sstream>
#include <limits>
#include <list>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include <exception>

using std::string;
using std::map;
using std::pair;
using std::vector;
using std::list;

// caffe.proto > LayerParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

namespace caffe {

template <typename Dtype>
void* FastWindowDataLayerPrefetch(void* layer_pointer) {

//#define DUMP_IMAGES
#ifdef DUMP_IMAGES
  static int iter_num = 0;
#endif

  FastWindowDataLayer<Dtype>* layer =
      reinterpret_cast<FastWindowDataLayer<Dtype>*>(layer_pointer);

  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  Dtype* top_overlap = layer->prefetch_overlap_->mutable_cpu_data();
  const Dtype scale = layer->layer_param_.fast_window_data_param().scale();
  const int batch_size = layer->layer_param_.fast_window_data_param().batch_size();
  const int crop_size = layer->layer_param_.fast_window_data_param().crop_size();
  const int context_pad = layer->layer_param_.fast_window_data_param().context_pad();
  const bool mirror = layer->layer_param_.fast_window_data_param().mirror();
  const float fg_fraction = layer->layer_param_.fast_window_data_param().fg_fraction();
  const Dtype* mean = layer->data_mean_.cpu_data();
  const int mean_off = (layer->data_mean_.width() - crop_size) / 2;
  const int mean_width = layer->data_mean_.width();
  const int mean_height = layer->data_mean_.height();
  const bool buffer_all_images = layer->layer_param_.fast_window_data_param().buffer_all_images();
  const float ambiguous_fraction_in_pos = layer->layer_param_.fast_window_data_param().ambiguous_fraction_in_pos();
  cv::Size cv_crop_size(crop_size, crop_size);
  const string& crop_mode = layer->layer_param_.fast_window_data_param().crop_mode();

  bool use_square = (crop_mode == "square") ? true : false;

  // zero out batch
  memset(top_data, 0, sizeof(Dtype)*layer->prefetch_data_->count());

  int num_fg_tmp = static_cast<int>(static_cast<float>(batch_size)
	      * fg_fraction);
  int num_amb_tmp = int( float(num_fg_tmp) * std::max(.0f, ambiguous_fraction_in_pos) );

  enum {
	  NORMAL_SAMPLE = 0,
	  PAIRED_GT     = 1
  };

  int pos_mode_limit = NORMAL_SAMPLE;
  if (layer->pair_pos_with_groundtruth()) {
	  num_fg_tmp  = ((num_fg_tmp+1)/2)*2; // make it even
	  num_amb_tmp = ((num_amb_tmp+1)/2)*2;
	  pos_mode_limit = PAIRED_GT;
  }
  const int num_fg = num_fg_tmp, num_amb = num_amb_tmp;
  const int num_samples[3] = { batch_size - num_fg, num_fg-num_amb, num_amb };

  // buffer windows and files
  list< FastWindowDataLayer_Aux::raw_image_t > raw_image_buffer;
  vector< FastWindowDataLayer_Aux::raw_image_t* >& raw_image_ptr = layer->raw_image_ptr_;
  vector< vector<float> >& windows = layer->cur_windows_;
  vector< bool >& do_mirror_list = layer->do_mirror_list_;

  {
	  int itemid = 0;
	  for (int is_fg = 0; is_fg < 3; ++is_fg) {
			const int mode_limit = (is_fg)? (pos_mode_limit) : 0 ;
			for (int dummy = 0; dummy < num_samples[is_fg];) {
				bool do_mirror = false;
				for (int mode_idx = 0; mode_idx <= mode_limit;
						++mode_idx, ++dummy, ++itemid) {

					if ( itemid>=batch_size ) {
						LOG(FATAL) << "itemid exceed range";
					}

					if (mode_idx==NORMAL_SAMPLE) {
						vector<float>* window_ptr;
						switch (is_fg) {
						case 0:
						{
							window_ptr = &(layer->bg_windows_[rand() % layer->bg_windows_.size()]);
							break;
						}
						case 1:
						{
							window_ptr = &(layer->fg_windows_[rand() % layer->fg_windows_.size()]);
							break;
						}
						case 2:
						{
							window_ptr = &(layer->amb_windows_[rand() % layer->amb_windows_.size()]);
							break;
						}
						default:
							LOG(FATAL) << "Internal error : unrecognized is_fg" ;
						}
						const vector<float>& window = *window_ptr;

						// windows[itemid] = window;
						memcpy( &(windows[itemid][0]), &(window[0]),
								sizeof(float) * (FastWindowDataLayer<Dtype>::NUM) );

						if (mirror && rand() % 2) {
							do_mirror = true;
						}

						pair<FastWindowDataLayer_Aux::x_image_t, vector<int> >& image =
								layer->image_database_[window[FastWindowDataLayer<Dtype>::IMAGE_INDEX]];
						if (image.first.second.empty()) {

							raw_image_buffer.push_back( FastWindowDataLayer_Aux::raw_image_t() );

							std::ifstream in_file( image.first.first.c_str(),
									std::ios_base::in | std::ios_base::binary );
							in_file.seekg( 0, std::ios_base::end );
							size_t file_size = in_file.tellg();
							in_file.seekg( 0, std::ios_base::beg );

							raw_image_buffer.back().resize( file_size );

							char* data_ptr = &(raw_image_buffer.back()[0]);
							in_file.read( data_ptr, file_size );

							raw_image_ptr[itemid] = &(raw_image_buffer.back());

						} else {
							raw_image_ptr[itemid] = &(image.first.second);
						}

					}

					do_mirror_list[itemid] = do_mirror;

				}
			}
	  }
  }

  /*
  // decode all the images
  vector< cv::Mat > cv_images(batch_size);
  {
	  #pragma omp parallel for
	  for ( int k = 0; k<batch_size; ++k ) {
		  if ( raw_image_ptr[k] != NULL) {
              cv_images[k] = cv::imdecode(*(raw_image_ptr[k]),
						CV_LOAD_IMAGE_COLOR);
		  }
	  }
  }
  */

  // sample from bg set then fg set
  //int opencv_threads_num = cv::getNumThreads();
  //cv::setNumThreads(1);

  //#pragma omp parallel for
  for ( int itemid0 = 0; itemid0<batch_size; ++itemid0 ) {
	  try {
        const bool is_fg = (itemid0>=num_samples[0]);
        const int mode_limit = (is_fg)? (pos_mode_limit) : 0 ;
        const int dummy = (is_fg)? (itemid0-num_samples[0]):(itemid0);
        const int mode_idx0 = (is_fg)? ((dummy%(mode_limit+1))):(0);
        /*
        LOG(INFO) << is_fg << "\t"
            << mode_limit << "\t"
            << dummy << "\t"
            << mode_idx0; */
        if ( mode_idx0 > 0 )
            continue;
        int itemid = itemid0;

        bool do_mirror = false;
        cv::Mat cv_img;
        const vector<float>* window_ptr;
        for (int mode_idx = 0; mode_idx <= mode_limit;
                ++mode_idx, ++itemid) {

			if ( itemid>=batch_size ) {
				LOG(FATAL) << "itemid exceed range";
			}


            if (mode_idx==NORMAL_SAMPLE) {
                // sample a window
            	window_ptr = &(windows[itemid]);

                // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
                do_mirror = do_mirror_list[itemid];

                // load the image containing the window
                pair<FastWindowDataLayer_Aux::x_image_t, vector<int> >& image =
                        layer->image_database_[(*window_ptr)[FastWindowDataLayer<Dtype>::IMAGE_INDEX]];

                // cv_img = cv_images[itemid];
                cv_img = cv::imdecode(*(raw_image_ptr[itemid]), CV_LOAD_IMAGE_COLOR );

                if (!cv_img.data) {
                    LOG(FATAL)<< "Could not open or find file " << image.first.first;
                    //return reinterpret_cast<void*>(NULL);
                }
            }
            const vector<float>& window = *window_ptr;

            const int channels = cv_img.channels();

            // crop window out of image and warp it
            int x1, y1, x2, y2;
            Dtype cur_label, cur_overlap;
            if (mode_idx==NORMAL_SAMPLE) {
                x1 = window[FastWindowDataLayer<Dtype>::X1];
                y1 = window[FastWindowDataLayer<Dtype>::Y1];
                x2 = window[FastWindowDataLayer<Dtype>::X2];
                y2 = window[FastWindowDataLayer<Dtype>::Y2];
                cur_label = window[FastWindowDataLayer<Dtype>::LABEL];
                cur_overlap = window[FastWindowDataLayer<Dtype>::OVERLAP];
            } else { // PAIRED_GT
                bool gt_availability = !(
                        isnan( window[FastWindowDataLayer<Dtype>::GX1] ) ||
                        isnan( window[FastWindowDataLayer<Dtype>::GY1] ) ||
                        isnan( window[FastWindowDataLayer<Dtype>::GX2] ) ||
                        isnan( window[FastWindowDataLayer<Dtype>::GY2] ) );
                CHECK(gt_availability) << "gt box is not available";

                x1 = window[FastWindowDataLayer<Dtype>::GX1];
                y1 = window[FastWindowDataLayer<Dtype>::GY1];
                x2 = window[FastWindowDataLayer<Dtype>::GX2];
                y2 = window[FastWindowDataLayer<Dtype>::GY2];
                /*LOG(INFO) << "item: " << itemid << "\t" << x1 << "\t" << y1 << "\t"
                    << x2 << "\t" << y2;*/
                cur_label = -std::abs( window[FastWindowDataLayer<Dtype>::LABEL] );
                cur_overlap = Dtype(1.);
                // ^^ always use neg label so that it will be ignored by regular svm_loss
            }

            // LOG(INFO) << "item: " << itemid << "\t" << cur_label << "\t" << cur_overlap;

            int pad_w = 0;
            int pad_h = 0;
            if (context_pad > 0 || use_square) {
                // scale factor by which to expand the original region
                // such that after warping the expanded region to crop_size x crop_size
                // there's exactly context_pad amount of padding on each side
                Dtype context_scale = static_cast<Dtype>(crop_size)
                        / static_cast<Dtype>(crop_size - 2 * context_pad);

                // compute the expanded region
                Dtype half_height = static_cast<Dtype>(y2 - y1 + 1) / 2.0;
                Dtype half_width = static_cast<Dtype>(x2 - x1 + 1) / 2.0;
                Dtype center_x = static_cast<Dtype>(x1) + half_width;
                Dtype center_y = static_cast<Dtype>(y1) + half_height;
                if (use_square) {
                    if (half_height > half_width) {
                        half_width = half_height;
                    } else {
                        half_height = half_width;
                    }
                }
                x1 = static_cast<int>(round(
                        center_x - half_width * context_scale));
                x2 = static_cast<int>(round(
                        center_x + half_width * context_scale));
                y1 = static_cast<int>(round(
                        center_y - half_height * context_scale));
                y2 = static_cast<int>(round(
                        center_y + half_height * context_scale));

                // the expanded region may go outside of the image
                // so we compute the clipped (expanded) region and keep track of
                // the extent beyond the image
                int unclipped_height = y2 - y1 + 1;
                int unclipped_width = x2 - x1 + 1;
                int pad_x1 = std::max(0, -x1);
                int pad_y1 = std::max(0, -y1);
                int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
                int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
                // clip bounds
                x1 = x1 + pad_x1;
                x2 = x2 - pad_x2;
                y1 = y1 + pad_y1;
                y2 = y2 - pad_y2;
                CHECK_GT(x1, -1);
                CHECK_GT(y1, -1);
                CHECK_LT(x2, cv_img.cols);
                CHECK_LT(y2, cv_img.rows);

                int clipped_height = y2 - y1 + 1;
                int clipped_width = x2 - x1 + 1;

                // scale factors that would be used to warp the unclipped
                // expanded region
                Dtype scale_x = static_cast<Dtype>(crop_size)
                        / static_cast<Dtype>(unclipped_width);
                Dtype scale_y = static_cast<Dtype>(crop_size)
                        / static_cast<Dtype>(unclipped_height);

                // size to warp the clipped expanded region to
                cv_crop_size.width = static_cast<int>(round(
                        static_cast<Dtype>(clipped_width) * scale_x));
                cv_crop_size.height = static_cast<int>(round(
                        static_cast<Dtype>(clipped_height) * scale_y));
                pad_x1 = static_cast<int>(round(
                        static_cast<Dtype>(pad_x1) * scale_x));
                pad_x2 = static_cast<int>(round(
                        static_cast<Dtype>(pad_x2) * scale_x));
                pad_y1 = static_cast<int>(round(
                        static_cast<Dtype>(pad_y1) * scale_y));
                pad_y2 = static_cast<int>(round(
                        static_cast<Dtype>(pad_y2) * scale_y));

                pad_h = pad_y1;
                // if we're mirroring, we mirror the padding too (to be pedantic)
                if (do_mirror) {
                    pad_w = pad_x2;
                } else {
                    pad_w = pad_x1;
                }

                // ensure that the warped, clipped region plus the padding
                // fits in the crop_size x crop_size image (it might not due to rounding)
                if (pad_h + cv_crop_size.height > crop_size) {
                    cv_crop_size.height = crop_size - pad_h;
                }
                if (pad_w + cv_crop_size.width > crop_size) {
                    cv_crop_size.width = crop_size - pad_w;
                }
            }

            cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
            cv::Mat cv_cropped_img = cv_img(roi);
            cv::resize(cv_cropped_img, cv_cropped_img, cv_crop_size, 0, 0,
                    cv::INTER_LINEAR);

            // horizontal flip at random
            if (do_mirror) {
                cv::flip(cv_cropped_img, cv_cropped_img, 1);
            }


#ifdef DUMP_IMAGES
            {
                char fn[256];
                sprintf(fn, "%d-%03d.png", iter_num, itemid);
                cv::imwrite(std::string(fn),cv_cropped_img);
            }
#endif
            // copy the warped window into top_data
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < cv_cropped_img.rows; ++h) {
                    for (int w = 0; w < cv_cropped_img.cols; ++w) {
                        Dtype pixel = static_cast<Dtype>(cv_cropped_img.at<
                                cv::Vec3b>(h, w)[c]);

                        top_data[((itemid * channels + c) * crop_size + h
                                + pad_h) * crop_size + w + pad_w] = (pixel
                                - mean[(c * mean_height + h + mean_off
                                        + pad_h) * mean_width + w + mean_off
                                        + pad_w]) * scale;
                    }
                }
            }

            // get window label & overlap
            top_label[itemid]   = cur_label;
            top_overlap[itemid] = cur_overlap;

#if 0
            // useful debugging code for dumping transformed windows to disk
            string file_id;
            std::stringstream ss;
            // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
            ss << rand();
            ss >> file_id;
            std::ofstream inf((string("dump/") + file_id +
                            string("_info.txt")).c_str(), std::ofstream::out);
            inf << layer->image_database_[window[FastWindowDataLayer<Dtype>::IMAGE_INDEX]].first.first << std::endl
            << window[FastWindowDataLayer<Dtype>::X1]+1 << std::endl
            << window[FastWindowDataLayer<Dtype>::Y1]+1 << std::endl
            << window[FastWindowDataLayer<Dtype>::X2]+1 << std::endl
            << window[FastWindowDataLayer<Dtype>::Y2]+1 << std::endl
            << do_mirror << std::endl
            << top_label[itemid] << std::endl
            << is_fg << std::endl;
            inf.close();
            std::ofstream top_data_file((string("dump/") + file_id +
                            string("_data.txt")).c_str(),
                    std::ofstream::out | std::ofstream::binary);
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < crop_size; ++h) {
                    for (int w = 0; w < crop_size; ++w) {
                        top_data_file.write(reinterpret_cast<char*>(
                                        &top_data[((itemid * channels + c) * crop_size + h)
                                        * crop_size + w]),
                                sizeof(Dtype));
                    }
                }
            }
            top_data_file.close();
#endif

        }
	  } catch (const std::exception& e) {
		  LOG(FATAL) << "error occurs when loading data : " << e.what();
	  }
  }
  //cv::setNumThreads(opencv_threads_num);
#ifdef DUMP_IMAGES
  ++iter_num;
#endif

  return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
FastWindowDataLayer<Dtype>::~FastWindowDataLayer<Dtype>() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
void FastWindowDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // SetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  CHECK_EQ(bottom.size(), 0) << "Window data Layer takes no input blobs.";
  CHECK_GE(top->size(), 2) << "Window data Layer prodcues two or three blobs as output.";
  CHECK_LE(top->size(), 3) << "Window data Layer prodcues two or three blobs as output.";

  const float ambiguous_fraction_in_pos = this->layer_param_.fast_window_data_param().ambiguous_fraction_in_pos();

  output_overlap_ = (top->size()>2);
  pair_pos_with_groundtruth_ = this->layer_param_.fast_window_data_param().pair_pos_with_groundtruth();
  include_ambiguous_pos_     = this->layer_param_.fast_window_data_param().include_ambiguous_pos();

  if ( !include_ambiguous_pos_ ) {
	  CHECK_LT( ambiguous_fraction_in_pos, 0 ) << "ambiguous_fraction_in_pos must be less than 0 when include_ambiguous_pos is not set";
  }

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2

  LOG(INFO) << "Window data layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.fast_window_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.fast_window_data_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.fast_window_data_param().fg_fraction() << std::endl
      << "  ambiguous sampling fraction in foreground: "
      << this->layer_param_.fast_window_data_param().ambiguous_fraction_in_pos();

  std::ifstream infile(this->layer_param_.fast_window_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.fast_window_data_param().source() << std::endl;

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  string hashtag;
  int image_index, channels;
  while (infile >> hashtag >> image_index) {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    channels = image_size[0];
    {
        FastWindowDataLayer_Aux::x_image_t ximg;
    	ximg.first = image_path;
    	image_database_.push_back(std::make_pair(ximg, image_size));
    }
    {
        FastWindowDataLayer_Aux::x_image_t& ximg = image_database_.back().first;
    	bool buffer_all_images = true;
    	if ( buffer_all_images ) {
            std::ifstream in_file( ximg.first.c_str(), 
                    std::ios_base::in | std::ios_base::binary );
    		in_file.seekg( 0, std::ios_base::end );
    		size_t file_size = in_file.tellg();
    		in_file.seekg( 0, std::ios_base::beg );
    		ximg.second.resize( file_size );
    		char* data_ptr = (&ximg.second[0]);
    		in_file.read( data_ptr, file_size );
    	}
    }

    // read each box
    int num_windows;
    infile >> num_windows;
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2, gx1, gy1, gx2, gy2;
      float overlap;
      std::string tline;
      {
          float x1_, y1_, x2_, y2_, gx1_, gy1_, gx2_, gy2_;
    	  shared_ptr<std::stringstream> line_in;
    	  do {
    		  std::getline( infile, tline );
			  line_in.reset( new std::stringstream(tline, std::ios_base::in ) );
			  (*line_in) >> label >> overlap >> x1_ >> y1_ >> x2_ >> y2_;
    	  } while ( line_in->fail() );
    	  (*line_in) >> gx1_ >> gy1_ >> gx2_ >> gy2_;
    	  if (line_in->fail()) {
    		  gx1_ = gy1_ = gx2_ = gy2_ = std::numeric_limits<Dtype>::quiet_NaN();
    	  }
          x1 = int(x1_); y1 = int(y1_); x2 = int(x2_); y2 = int(y2_);
          gx1 = int(gx1_); gy1 = int(gy1_); gx2 = int(gx2_); gy2 = int(gy2_);
      }

      vector<float> window(FastWindowDataLayer::NUM);
      window[FastWindowDataLayer::IMAGE_INDEX] = image_index;
      window[FastWindowDataLayer::LABEL] = label;
      window[FastWindowDataLayer::OVERLAP] = overlap;
      window[FastWindowDataLayer::X1] = x1;
      window[FastWindowDataLayer::Y1] = y1;
      window[FastWindowDataLayer::X2] = x2;
      window[FastWindowDataLayer::Y2] = y2;
      window[FastWindowDataLayer::GX1] = gx1;
      window[FastWindowDataLayer::GY1] = gy1;
      window[FastWindowDataLayer::GX2] = gx2;
      window[FastWindowDataLayer::GY2] = gy2;


      // add window to foreground list or background list

      if (overlap < this->layer_param_.fast_window_data_param().bg_threshold() || overlap <= float(0.)) {
        // background window, force label and overlap to 0
        window[FastWindowDataLayer::LABEL] = 0;
        window[FastWindowDataLayer::OVERLAP] = 0;
        bg_windows_.push_back(window);
        label_hist[0]++;
      } else {
    	  bool is_clear_pos = (overlap >= this->layer_param_.fast_window_data_param().fg_threshold());
    	  if( is_clear_pos || include_ambiguous_pos_ ) {
        	  vector<vector<float> >* target_windows_ptr;
    		  int label = window[FastWindowDataLayer::LABEL];
    		  CHECK_GT(label, 0);
    		  if ( is_clear_pos ) {
    			  target_windows_ptr = &fg_windows_;
    		  } else {
    			  window[FastWindowDataLayer::LABEL] = -label;
    			  if (ambiguous_fraction_in_pos<0)	//merge amb into pos (for backward compatibility)
    				  target_windows_ptr = &fg_windows_;
    			  else
    				  target_windows_ptr = &amb_windows_;
    		  }
    		  target_windows_ptr->push_back(window);
    		  label_hist.insert(std::make_pair(label, 0));
    		  label_hist[label]++;
		  }
      }
    }

    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  }

  LOG(INFO) << "Number of images: " << image_index+1;

  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }

  LOG(INFO) << "Neg: " << bg_windows_.size();
  LOG(INFO) << "Pos: " << fg_windows_.size();
  LOG(INFO) << "Amb: " << amb_windows_.size();

  LOG(INFO) << "Amount of context padding: "
      << this->layer_param_.fast_window_data_param().context_pad();

  LOG(INFO) << "Crop mode: " << this->layer_param_.fast_window_data_param().crop_mode();

  // image
  int crop_size = this->layer_param_.fast_window_data_param().crop_size();
  CHECK_GT(crop_size, 0);
  (*top)[0]->Reshape(
      this->layer_param_.fast_window_data_param().batch_size(), channels, crop_size, crop_size);
  prefetch_data_.reset(new Blob<Dtype>(
      this->layer_param_.fast_window_data_param().batch_size(), channels, crop_size, crop_size));

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(this->layer_param_.fast_window_data_param().batch_size(), 1, 1, 1);
  prefetch_label_.reset(
      new Blob<Dtype>(this->layer_param_.fast_window_data_param().batch_size(), 1, 1, 1));
  // overlap
  if (output_overlap_) {
	  (*top)[2]->Reshape(this->layer_param_.fast_window_data_param().batch_size(), 1, 1, 1);
  }
  prefetch_overlap_.reset(
      new Blob<Dtype>(this->layer_param_.fast_window_data_param().batch_size(), 1, 1, 1));

  // buffers
  {
	  raw_image_ptr_.resize(this->layer_param_.fast_window_data_param().batch_size(), NULL);
	  cur_windows_.resize(this->layer_param_.fast_window_data_param().batch_size(), vector<float>(FastWindowDataLayer::NUM) );
	  do_mirror_list_.resize(this->layer_param_.fast_window_data_param().batch_size());
  }

  // check if we want to have mean
  if (this->layer_param_.fast_window_data_param().has_mean_file()) {
    BlobProto blob_proto;
    LOG(INFO) << "Loading mean file from" << this->layer_param_.fast_window_data_param().mean_file();
    ReadProtoFromBinaryFile(this->layer_param_.fast_window_data_param().mean_file().c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.width(), data_mean_.height());
    CHECK_EQ(data_mean_.channels(), channels);
  } else {
    // Simply initialize an all-empty mean.
		data_mean_.Reshape(1, channels, crop_size, crop_size);

		Dtype mean_value[channels];

		int m_channels = this->layer_param_.fast_window_data_param().mean_values_size();

		for (int i=0; i<channels; ++i) {
			if (m_channels>i)
				mean_value[i] = this->layer_param_.fast_window_data_param().mean_values(i);
			else
				mean_value[i] = this->layer_param_.fast_window_data_param().mean_value();
		}

  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CHECK(!pthread_create(&thread_, NULL, FastWindowDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
Dtype FastWindowDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
      sizeof(Dtype) * prefetch_data_->count());
  memcpy((*top)[1]->mutable_cpu_data(), prefetch_label_->cpu_data(),
      sizeof(Dtype) * prefetch_label_->count());
  if (output_overlap_) {
	  memcpy((*top)[2]->mutable_cpu_data(), prefetch_overlap_->cpu_data(),
	      sizeof(Dtype) * prefetch_overlap_->count());
  }
  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, FastWindowDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  return Dtype(0.);
}

INSTANTIATE_CLASS(FastWindowDataLayer);

}  // namespace caffe
