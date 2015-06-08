// Copyright 2013 Yangqing Jia
#include <glog/logging.h>
#include <stdint.h>
#include "caffe/util/fake_leveldb.hpp"

#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using caffe::Datum;
using caffe::BlobProto;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 3 || argc > 4 ) {
    LOG(ERROR) << "Usage: demo_compute_image_mean input_leveldb output_file [grayscale/color]";
    return(0);
  }

  boost::shared_ptr< caffe_modified::fake_leveldb_imagedb > db;
  db.reset( new caffe_modified::fake_leveldb_imagedb );

  LOG(INFO) << "Opening fake leveldb " << argv[1];
  int status = db->OpenReadOnly( argv[1] );

  CHECK(~status) << "Failed to open leveldb " << argv[1] << std::endl
		  << "Error code" << status << std::endl ;

  int color_type = caffe_modified::fake_leveldb::BGR_COLOR;
  if ( argc>=4 ) {
      std::string color_type_str = argv[3];
      if ( color_type_str == "grayscale" ) {
          color_type = caffe_modified::fake_leveldb::GRAY_SCALE;
      } else if ( color_type_str == "color" ) {
          color_type = caffe_modified::fake_leveldb::BGR_COLOR;
      } else {
          LOG(ERROR) << "Unrecognized color type : " << color_type_str  ;
      }
  }
  
  boost::shared_ptr<caffe_modified::fake_leveldb_imagedb::Iterator> 
      it( db->NewIterator(2,0,color_type) );
  it->SeekToFirst();
  Datum datum;
  BlobProto sum_blob;
  int count = 0;
  std::pair<cv::Mat,int> ldata = it->get_labeled_image();
  ReadImageToDatum( ldata.first, ldata.second, 0,0, &datum );
  sum_blob.set_num(1);
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  for (int i = 0; i < datum.data().size(); ++i) {
    sum_blob.add_data(0.);
  }
  LOG(INFO) << "Starting Iteration";
  for (it->SeekToFirst(); it->Valid(); it->Next()) {

	std::pair<cv::Mat,int> ldata = it->get_labeled_image();
	ReadImageToDatum( ldata.first, ldata.second, 0,0, &datum );
    const std::string& data = datum.data();
    CHECK_EQ(data.size(), data_size) << "Incorrect data field size " << data.size();
    for (int i = 0; i < data.size(); ++i) {
      sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  LOG(INFO) << "Write to " << argv[2];
  WriteProtoToBinaryFile(sum_blob, argv[2]);


  return 0;
}
