/*
 * fake_leveldb.h
 *
 *  Created on: Dec 29, 2013
 *      Author: zhangyuting
 *
 *  Use a image list file to mimic the behavior of leveldb
 *  for image dataset
 */


#ifndef CAFFE_MODIFIED_FAKE_LEVELDB_H_
#define CAFFE_MODIFIED_FAKE_LEVELDB_H_

#include <string>
#include <vector>
#include <limits>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>
#include <opencv2/opencv.hpp>

namespace caffe_modified {

struct size2d {
	int w;
	int h;
	size2d() : w(0), h(0) {}
	size2d( int s ) : w(s), h(s) {}
	size2d( int w, int h ) : w(w), h(h) {}
};

class _fake_leveldb_iterator {
public:
	virtual void SeekToFirst() = 0;
	virtual void Next() = 0;
	virtual bool Valid() = 0;
	virtual std::pair<cv::Mat,int> get_labeled_image() = 0;
	virtual ~_fake_leveldb_iterator() {}
};

class fake_leveldb : public boost::noncopyable {
public:
	typedef _fake_leveldb_iterator  Iterator;
	virtual Iterator* NewIterator( size_t max_open_files, size2d canonical_size, int color_type ) = 0;
	virtual int OpenReadOnly( const std::string& database_path ) = 0;
	virtual ~fake_leveldb() {};

	enum {
		AS_IS_COLOR = 0,
		BGR_COLOR   = 1,
		GRAY_SCALE  = 2
	};
};

// --------------------------------------------- fake_leveldb_imagedb

struct _fake_leveldb_imagedb_single_file_range {
	size_t start_loc;
	size_t length;

    static size_t MAX_POSSIBLE_LENGTH() { return std::numeric_limits<size_t>::max(); }

	_fake_leveldb_imagedb_single_file_range() : start_loc(0), length( MAX_POSSIBLE_LENGTH() ) {}
	_fake_leveldb_imagedb_single_file_range(size_t start_loc_, size_t length_) :
		start_loc(start_loc_), length( length_ ) {}
};

struct _fake_leveldb_imagedb_single_fileinfo : public _fake_leveldb_imagedb_single_file_range {
	std::string filename;
	_fake_leveldb_imagedb_single_fileinfo()  {}
	_fake_leveldb_imagedb_single_fileinfo( const std::string& filename_ ) :
		filename(filename_) {}
	_fake_leveldb_imagedb_single_fileinfo( 
        const std::string& filename_, size_t start_loc_, size_t length_ ) :
		_fake_leveldb_imagedb_single_file_range(start_loc_, length_), filename(filename_) {}
};

struct _fake_leveldb_imagedb_iterator_inner;

class _fake_leveldb_imagedb_iterator : public _fake_leveldb_iterator {
	class _fake_leveldb_imagedb_iterator_inner* _h;
public:

    typedef _fake_leveldb_imagedb_single_file_range single_file_range;
    typedef _fake_leveldb_imagedb_single_fileinfo   single_fileinfo;
    typedef std::pair<single_fileinfo, int>   labeled_data_t;
    typedef std::vector< labeled_data_t > labeled_data_vec_t;

	_fake_leveldb_imagedb_iterator( const std::string& root_dir,
			const labeled_data_vec_t& lines,
			size_t max_open_files, size2d canonical_size, int color_type );
	_fake_leveldb_imagedb_iterator( const _fake_leveldb_imagedb_iterator& _r );
	_fake_leveldb_imagedb_iterator& operator = (
			const _fake_leveldb_imagedb_iterator& _r );
	~_fake_leveldb_imagedb_iterator();

	void SeekToFirst();
	void Next();
	bool Valid();
	std::pair<cv::Mat,int> get_labeled_image();
};


class fake_leveldb_imagedb : public fake_leveldb {
public:

    typedef _fake_leveldb_imagedb_single_file_range single_file_range;
    typedef _fake_leveldb_imagedb_single_fileinfo   single_fileinfo;

	typedef std::pair<single_fileinfo, int>   labeled_data_t;
	typedef std::vector< labeled_data_t > labeled_data_vec_t;
private:
	boost::shared_ptr<labeled_data_vec_t> lines_;
	boost::shared_ptr<std::string>        root_dir_;
public:
	typedef _fake_leveldb_imagedb_iterator  Iterator;

	Iterator* NewIterator( size_t max_open_files = 2, size2d canonical_size = 0, int color_type = BGR_COLOR );

	int OpenReadOnly( const std::string& database_path );

};


// --------------------------------------------- fake_leveldb_bbox


struct _fake_leveldb_bbox_single_bbox_range {
	int x1, y1, x2, y2;

    static size_t MAX_POSSIBLE_LENGTH() { return std::numeric_limits<size_t>::max(); }

    _fake_leveldb_bbox_single_bbox_range() {}
    _fake_leveldb_bbox_single_bbox_range( int x1_, int y1_, int x2_, int y2_ ) :
		x1(x1_), y1(y1_), x2(x2_), y2(y2_) {}
};

struct _fake_leveldb_bbox_single_bboxinfo : public _fake_leveldb_bbox_single_bbox_range {
	std::string filename;
	_fake_leveldb_bbox_single_bboxinfo()  {}
	_fake_leveldb_bbox_single_bboxinfo( const std::string& filename_ ) :
		filename(filename_) {}
	_fake_leveldb_bbox_single_bboxinfo(
        const std::string& filename_, int x1_, int y1_, int x2_, int y2_ ) :
        	_fake_leveldb_bbox_single_bbox_range(x1_,y1_,x2_,y2_), filename(filename_) {}
};



struct _fake_leveldb_bbox_iterator_inner;

class _fake_leveldb_bbox_iterator : public _fake_leveldb_iterator {
	class _fake_leveldb_bbox_iterator_inner* _h;
public:

    typedef _fake_leveldb_bbox_single_bbox_range single_bbox_range;
    typedef _fake_leveldb_bbox_single_bboxinfo   single_bboxinfo;
    typedef std::pair<single_bboxinfo, int>   labeled_data_t;
    typedef std::vector< labeled_data_t > labeled_data_vec_t;

	_fake_leveldb_bbox_iterator( const std::string& root_dir,
			const labeled_data_vec_t& lines,
			size_t max_open_files, size2d canonical_size, int color_type  );
	_fake_leveldb_bbox_iterator( const _fake_leveldb_bbox_iterator& _r );
	_fake_leveldb_bbox_iterator& operator = (
			const _fake_leveldb_bbox_iterator& _r );
	~_fake_leveldb_bbox_iterator();

	void SeekToFirst();
	void Next();
	bool Valid();
	std::pair<cv::Mat,int> get_labeled_image();
};


class fake_leveldb_bbox : public fake_leveldb {
public:

    typedef _fake_leveldb_bbox_single_bbox_range single_bbox_range;
    typedef _fake_leveldb_bbox_single_bboxinfo   single_bboxinfo;
	typedef std::pair<single_bboxinfo, int>   labeled_data_t;
	typedef std::vector< labeled_data_t > labeled_data_vec_t;
private:
	boost::shared_ptr<labeled_data_vec_t> lines_;
	boost::shared_ptr<std::string>        root_dir_;
public:
	typedef _fake_leveldb_bbox_iterator  Iterator;

	Iterator* NewIterator( size_t max_open_files = 20, size2d canonical_size = 0, int color_type = BGR_COLOR  );

	int OpenReadOnly( const std::string& database_path );

};


}


#endif


