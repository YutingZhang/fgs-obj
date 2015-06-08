/*
 * restaurant_scheduler.h
 *
 *  Created on: Dec 29, 2013
 *      Author: zhangyuting
 */

#ifndef RESTAURANT_SCHEDULER_HPP_
#define RESTAURANT_SCHEDULER_HPP_

#include <cstddef>
#include <limits>
#include <memory>
#include <boost/noncopyable.hpp>

#define RESTAURANT_SCHEDULER 1	// VERSION 1

namespace zeta {

#if __cplusplus >= 201103L
#define __smart_ptr std::unique_ptr
#else
#define __smart_ptr std::auto_ptr
#endif


template <class INPUT_T, class OUTPUT_T>
class single_worker {
public:
	typedef single_worker<INPUT_T,OUTPUT_T> self_t;
	virtual OUTPUT_T operator () ( INPUT_T ) = 0;
	virtual self_t* clone() const = 0;
	virtual ~single_worker() {}
};

class __single_worker_wrapper_base : public boost::noncopyable {
public:
	virtual void* operator () ( void* ) = 0;
	virtual __single_worker_wrapper_base* clone() const = 0;
	virtual ~__single_worker_wrapper_base() {}
};

template <class INPUT_T, class OUTPUT_T>
class __single_worker_wrapper : public __single_worker_wrapper_base {
public:
	typedef single_worker<INPUT_T,OUTPUT_T> worker_t;
	typedef __single_worker_wrapper<INPUT_T,OUTPUT_T> self_t;
private:
	worker_t* __worker;
public:
	__single_worker_wrapper( const worker_t& worker ) : __worker(worker.clone())  {}
	virtual void* operator () ( void* input_ptr ) {
		INPUT_T  a = *(reinterpret_cast<INPUT_T*>(input_ptr));
		OUTPUT_T b = (*__worker)(a);
		OUTPUT_T* bp = new OUTPUT_T(b);
		return bp;
	}
	virtual self_t* clone() const {
		return new self_t( *__worker );
	}
	virtual ~__single_worker_wrapper() { delete __worker; }
};

class __type_concealed_handler : public boost::noncopyable {
public:
	virtual void delete_object( void* ) const = 0;
	virtual __type_concealed_handler* clone() const = 0;
	virtual ~__type_concealed_handler() {}
};

template<class T>
class __type_concealed_handler_with_type : public __type_concealed_handler {
public:
	virtual void delete_object( void* p ) const { delete reinterpret_cast<T*> (p); }
	virtual __type_concealed_handler_with_type* clone() const {
		return new __type_concealed_handler_with_type();
	}
	virtual ~__type_concealed_handler_with_type() {}
};


class __restaurant_scheduler_base_inner;

class __restaurant_scheduler_base : public boost::noncopyable {
//	friend class __restaurant_scheduler_base_inner;
public:
	typedef size_t	key_t;

	static size_t AUTO_WORKER_NUM() { return 0; }
	static int HIGHEST_PRIORITY() { return std::numeric_limits<int>::min(); }
	static int IDLE_PRIORITY() { return (std::numeric_limits<int>::max()-1); }	// you can't save a task from idle
	static int TO_BE_DECALIMED_PRIORITY() { return std::numeric_limits<int>::max(); }	// you can't save a task from idle
	static key_t NULL_KEY() { return std::numeric_limits<key_t>::max(); }

private:
	__restaurant_scheduler_base_inner* __h;
protected:
	key_t  _claim_key( void* in_ptr, int priority );
	void  _change_priority( key_t key, int priority );
	void  _disclaim_key(  key_t key );
	void* _in_ptr( key_t key );
	void* _out_ptr( key_t key );
	void  _wait_key( key_t key );
public:
	typedef void* token_t;

	token_t stop_processing();
	bool resume_processing( token_t authorized_token );	//true - OK; false - wrong token
	void wait_for_stopped();
	void wait_for_resumed();
public:
	size_t worker_num() const;
	void set_worker_num( size_t new_worker_num );
public:
	explicit __restaurant_scheduler_base(
			const __type_concealed_handler& input_type_concealer,
			const __type_concealed_handler& output_type_concealer,
			const __single_worker_wrapper_base& prototype_worker_wrapper,
			size_t worker_num = AUTO_WORKER_NUM() );
	virtual ~__restaurant_scheduler_base();
};

template <class INPUT_T, class OUTPUT_T>
class restaurant_scheduler : public __restaurant_scheduler_base {
public:
	using __restaurant_scheduler_base::key_t;
	typedef single_worker<INPUT_T, OUTPUT_T> worker;
public:
	explicit restaurant_scheduler(
			const worker& prototype_worker, size_t worker_num = AUTO_WORKER_NUM() );
	~restaurant_scheduler();
	// claim: you will need this work in the future
	key_t claim( INPUT_T input, int priority = 0 );
	// disclaim: you will NOT need this work (any obtained output will be released)
	void disclaim( key_t key );
	void change_priority( key_t key, int priority ) { this->_change_priority(key,priority); }
	void wait( key_t key )    { this->_wait_key(key); }
	OUTPUT_T read( key_t key );
	OUTPUT_T retrieve( key_t key );
};


template <class INPUT_T, class OUTPUT_T>
restaurant_scheduler<INPUT_T,OUTPUT_T>::restaurant_scheduler(
		const worker& prototype_worker, size_t worker_num ) :
		__restaurant_scheduler_base(
				__type_concealed_handler_with_type<INPUT_T>(),
				__type_concealed_handler_with_type<OUTPUT_T>(),
				__single_worker_wrapper<INPUT_T,OUTPUT_T>( prototype_worker ), worker_num ) {
}

template <class INPUT_T, class OUTPUT_T>
restaurant_scheduler<INPUT_T,OUTPUT_T>::~restaurant_scheduler() { }

template <class INPUT_T, class OUTPUT_T>
__restaurant_scheduler_base::key_t
restaurant_scheduler<INPUT_T,OUTPUT_T>::claim( INPUT_T input, int priority ) {
	INPUT_T* a = new INPUT_T(input);
	return this->_claim_key( a, priority );
}

template <class INPUT_T, class OUTPUT_T>
void restaurant_scheduler<INPUT_T,OUTPUT_T>::disclaim( key_t key ) {
	this->_disclaim_key( key );
}


template <class INPUT_T, class OUTPUT_T>
OUTPUT_T restaurant_scheduler<INPUT_T,OUTPUT_T>::read(
		__restaurant_scheduler_base::key_t key ) {
	this->_wait_key( key );
	OUTPUT_T* bp = reinterpret_cast<OUTPUT_T*>( _out_ptr(key) );
	return *bp;
}

template <class INPUT_T, class OUTPUT_T>
OUTPUT_T restaurant_scheduler<INPUT_T,OUTPUT_T>::retrieve(
		__restaurant_scheduler_base::key_t key ) {
	OUTPUT_T b = read(key);
	disclaim( key );
	return b;
}


}


#endif /* RESTAURANT_SCHEDULER_H_ */
