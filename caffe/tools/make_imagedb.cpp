// Copyright 2013 Yangqing Jia
// This program converts a set of images to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//    convert_dataset ROOTFOLDER LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....
// if the last argument is 1, a random shuffle will be carried out before we
// process the file lines.
// You are responsible for shuffling the files yourself.

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>

using std::pair;
using std::string;
using std::stringstream;

const size_t DEFAULT_ROUGH_FILE_SIZE = 100*1024*1024;

int main(int argc, char** argv) {
  size_t ROUGH_FILE_SIZE = DEFAULT_ROUGH_FILE_SIZE;
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 4) {
    LOG(ERROR) << "Usage: convert_imageset ROOTFOLDER LISTFILE DB_NAME [TDAT_FILE_SIZE]";
    return 0;
  }
  if (argc>=5) {
	  ROUGH_FILE_SIZE = boost::lexical_cast<size_t>(argv[4]);
	  CHECK( ROUGH_FILE_SIZE > 0 ) << "Wrong param for TDAT_FILE_SIZE";
}
  std::ifstream infile(argv[2]);
  std::vector< std::pair<string, int> > lines;
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  infile.close();

  string root_dir( argv[1] );
  string db_dir( argv[3] );
  std::system( ( "mkdir -p '"+db_dir+"'" ).c_str() );

  std::ofstream out_index_file( (db_dir+"/index.txt").c_str() );
  if ( out_index_file.fail() ) {
	    LOG(ERROR) << "Cannot write to " << db_dir+"/index.txt";
	    return 1;
  }

  boost::shared_ptr<std::ofstream> cur_outfile;

  size_t k = 0;

  size_t cur_file_size = ROUGH_FILE_SIZE;

  string tdat_filename;

  size_t i;
  for ( i = 0; i<lines.size(); ++i ) {

	  if ( cur_file_size>= ROUGH_FILE_SIZE ) {
		  ++k;
		  tdat_filename = boost::lexical_cast<std::string>(k) +	".tdat";
		  cur_outfile.reset( new std::ofstream( ( db_dir+"/" +
				  tdat_filename ).c_str(), std::ios_base::binary ) );
		  cur_file_size = 0;
		  if (i>0)
			  LOG(INFO) << "Close at : " << i-1;
		  LOG(INFO) << "Create tdat file : " << tdat_filename;
		  LOG(INFO) << "Open at : " << i;
	  }

	  // LOG(INFO) << "Image " << (i+1) << " / " << lines.size();

	  std::ifstream cur_infile( ( root_dir + "/" + lines[i].first ).c_str(), std::ios_base::binary );

	  cur_infile.seekg( 0, std::ios_base::end );
	  size_t cur_infile_size = cur_infile.tellg();
	  cur_infile.seekg( 0, std::ios_base::beg );

	  std::vector<char> C( cur_infile_size );
	  cur_infile.read( &(C[0]), cur_infile_size );

	  (*cur_outfile).write( &(C[0]), cur_infile_size );

	  out_index_file << tdat_filename << "\t"
			  << cur_file_size << "\t"
			  << cur_infile_size << "\t"
			  << lines[i].second << std::endl;

	  cur_file_size += cur_infile_size;

  }

  cur_outfile.reset();
  if (i>0)
	  LOG(INFO) << "Close at : " << i-1;

  out_index_file.close();

  return 0;
}
