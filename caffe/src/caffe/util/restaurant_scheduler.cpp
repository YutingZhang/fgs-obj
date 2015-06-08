/* Author: Yuting Zhang */


#include "caffe/util/zeta/scheduler/restaurant_scheduler.hpp"

#include <map>
#include <vector>
#include <list>
#include <cmath>

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>
#include <boost/function.hpp>

#include <iostream>
#include <limits>

namespace zeta {

// utils

template<class T>
void __resize_vector_with_default_constructor( std::vector<T>& v, size_t size ) {
	if ( size>v.size() ) {
		v.reserve( size );
		while( v.size()<size )
			v.push_back( T() );
	} else {
		v.resize( size );
	}

}


boost::mutex stdio_mutex;
#define COUT( A ) { boost::lock_guard<boost::mutex> stdio_lock(stdio_mutex); std::cout << A ; }


// -----------------------------------------------------------------------------

class auto_thread_pool;

struct __auto_thread_pool_worker_data {

	typedef boost::function<void ()> callable_t;

	boost::mutex working_mutex;
	boost::shared_ptr< boost::lock_guard<boost::mutex> > idle_lock;

	boost::mutex suicide_mutex;
	boost::shared_ptr< boost::lock_guard<boost::mutex> > alive_lock;


	boost::shared_ptr<boost::thread> my_thread;
	callable_t fun;

	__auto_thread_pool_worker_data() :
		idle_lock( new boost::lock_guard<boost::mutex>(working_mutex) ),
		alive_lock(new boost::lock_guard<boost::mutex>(suicide_mutex) ) {}

};

class __auto_thread_pool_worker {
public:
	typedef boost::function<void ()> callable_t;
private:
	auto_thread_pool& __p;
	__auto_thread_pool_worker_data& __d;
public:
	__auto_thread_pool_worker( auto_thread_pool& p, __auto_thread_pool_worker_data& d ) :
		__p(p), __d(d) { }
	void operator()();
};


class auto_thread_pool : public boost::noncopyable {
	friend class __auto_thread_pool_worker;
private:
	size_t worker_num;
	size_t max_worker_num;
	mutable boost::mutex mwn_mutex;
	mutable boost::mutex overflow_mutex;
	boost::shared_ptr< boost::lock_guard<boost::mutex> > overflow_lock;

	boost::shared_mutex processing_mutex;
	mutable boost::mutex list_mutex;
	mutable boost::mutex destructor_mutex;
	void give_back( __auto_thread_pool_worker_data& d );
private:
	typedef std::list< __auto_thread_pool_worker_data* > wd_list_t;
	wd_list_t available_workers;	// growth only
public:
	typedef boost::function<void ()> callable_t;

	void run( callable_t fun );	// if no thread available it will create new ones

	size_t current_worker_num() const;

	size_t max_allowed_worker_num() const;
	void   set_max_allowed_worker_num( size_t max_allowed_worker_num );
	static size_t MAX_POSSIBLE_WORKER_NUM() { return std::numeric_limits<size_t>::max(); }

	auto_thread_pool( size_t max_allowed_worker_num_ = MAX_POSSIBLE_WORKER_NUM() );
	~auto_thread_pool();
};

void __auto_thread_pool_worker::operator()() {
	for(;;) {
		boost::shared_ptr< boost::shared_lock<boost::shared_mutex> > processing_lock;
		{
			boost::lock_guard<boost::mutex> working_lock(__d.working_mutex);
			processing_lock.reset(new boost::shared_lock<boost::shared_mutex>(__p.processing_mutex) );
			{
				boost::unique_lock< boost::mutex > suicide_lock( __d.suicide_mutex, boost::try_to_lock );
				if (suicide_lock)
					break;
			}
			__d.fun();
		}
		__p.give_back( __d );
	}
}

auto_thread_pool::auto_thread_pool( size_t max_allowed_worker_num_ ) :
		worker_num(0), max_worker_num(max_allowed_worker_num_) {}

void auto_thread_pool::give_back( __auto_thread_pool_worker_data& d ) {
	d.fun.clear();
	d.idle_lock.reset( new boost::lock_guard<boost::mutex>(d.working_mutex) );
	{
		boost::lock_guard<boost::mutex> lock(list_mutex);
		available_workers.push_front( &d );

		overflow_lock.reset();
	}
}

size_t auto_thread_pool::max_allowed_worker_num() const {
	boost::lock_guard<boost::mutex> lock( mwn_mutex );
	return max_worker_num;
}

void auto_thread_pool::set_max_allowed_worker_num( size_t max_allowed_worker_num ) {
	boost::lock_guard<boost::mutex> lock( mwn_mutex );
	max_worker_num = max_allowed_worker_num;
}


void auto_thread_pool::run( auto_thread_pool::callable_t fun ) {

	boost::unique_lock<boost::mutex> no_destruction_lock( destructor_mutex, boost::try_to_lock );
	if (!no_destruction_lock) {	// give up if during destruction
		return;
	}

	boost::lock_guard<boost::mutex> lock( mwn_mutex ); // also block other attempts for run

	bool is_overflow;
	{
		boost::lock_guard<boost::mutex> lock( list_mutex );
		is_overflow = (worker_num>max_worker_num)
				|| (worker_num==max_worker_num && available_workers.empty());
	}
	if( is_overflow ) {
		for(;;) {
			{
				boost::lock_guard<boost::mutex> lock( list_mutex );

				if ( !overflow_lock.get() )
					overflow_lock.reset( new boost::lock_guard< boost::mutex >(overflow_mutex) );

				// clean unnecessary threads
				wd_list_t to_be_removed;
				while( worker_num>max_worker_num && available_workers.empty() ) {
					__auto_thread_pool_worker_data* d = available_workers.front();
					d->alive_lock.reset();
					d->idle_lock.reset();
					to_be_removed.push_back(d);
					available_workers.pop_front();
					--worker_num;
				}
				for( wd_list_t::iterator iter=to_be_removed.begin();
						iter != to_be_removed.end(); ++iter ) {
					delete (*iter);
					(*iter)->my_thread->join();
				}

				is_overflow = (worker_num>max_worker_num)
						|| (worker_num==max_worker_num && available_workers.empty());

			}

			if (!is_overflow)
				break;

			boost::lock_guard< boost::mutex > overflow_lock2(overflow_mutex);

		}
	}

	__auto_thread_pool_worker_data* cur;
	{
		// obtain a thread (either new or existing)
		boost::lock_guard<boost::mutex> lock(list_mutex);
		if (available_workers.empty()) {
			// new data and new thread
			__auto_thread_pool_worker_data* d = new __auto_thread_pool_worker_data();
			d->my_thread.reset( new boost::thread( __auto_thread_pool_worker(*this,*d) ) );
			available_workers.push_front( d );
			++worker_num;
		}
		cur = available_workers.back();
		available_workers.pop_back();
	}
	cur->fun = fun;
	cur->idle_lock.reset();
}

auto_thread_pool::~auto_thread_pool() {
	boost::lock_guard<boost::mutex> no_destruction_lock( destructor_mutex );
	// wait for all work done
	{
		boost::upgrade_lock<boost::shared_mutex> shared_processing_lock(processing_mutex);
		boost::upgrade_to_unique_lock<boost::shared_mutex> unique_processing_lock(shared_processing_lock);
	}
	// send stop signals
	for ( wd_list_t::iterator iter = available_workers.begin();
			iter != available_workers.end(); ++iter ) {
		(*iter)->alive_lock.reset();
		(*iter)->idle_lock.reset();
	}
	// join all thread & release all data
	for ( wd_list_t::iterator iter = available_workers.begin();
			iter != available_workers.end(); ++iter ) {
		(*iter)->my_thread->join();
		delete (*iter);
	}
}

size_t auto_thread_pool::current_worker_num() const {
	boost::lock_guard<boost::mutex> lock(list_mutex);
	return worker_num;
}

// ---------------------------------------------------------------



typedef __restaurant_scheduler_base_inner RSBI;
typedef __restaurant_scheduler_base RSB;

class RSBI_worker;
class RSBI_worker_pool;

class __restaurant_scheduler_base_inner {
	friend class RSBI_worker;
	friend class RSBI_worker_pool;
private:
	boost::shared_ptr<__type_concealed_handler> __input_c;
	boost::shared_ptr<__type_concealed_handler> __output_c;
	boost::shared_ptr<__single_worker_wrapper_base> __prototype_worker_wrapper;
public:
	typedef __restaurant_scheduler_base::key_t key_t;
	__restaurant_scheduler_base_inner(
			const __type_concealed_handler& input_type_concealer,
			const __type_concealed_handler& output_type_concealer,
			const __single_worker_wrapper_base& prototype_worker_wrapper,
			size_t worker_num_ );
	~__restaurant_scheduler_base_inner();
private:
	/// task list for different priority
	boost::mutex task_list_mutex;
	boost::mutex task_removal_mutex;
	std::map< int, std::list<key_t> > tasks;
	std::map< int, std::list<key_t> >::iterator idle_task_list_iter;
	std::map< int, std::list<key_t> >::iterator to_be_disclaimed_list_iter;
public:

	struct task_unit {
		int   priority;
		std::list<key_t>::iterator iter;
		boost::shared_ptr< boost::mutex > busy_mutex;
		boost::shared_ptr< boost::mutex > uncomplete_mutex;
		boost::shared_ptr< boost::lock_guard<boost::mutex> > uncomplete_lock;
		task_unit() :
			priority(0), busy_mutex( new boost::mutex ),
			uncomplete_mutex( new boost::mutex ),
			uncomplete_lock( new boost::lock_guard<boost::mutex>( *uncomplete_mutex ) ) { }
	};

	struct task_unit_wrapper {	// for the convenience for copying
		boost::shared_ptr<task_unit> p;
		task_unit_wrapper() : p( new task_unit ) {}
	};

private:
	void set_task( key_t key, void* input, int priority );
	void remove_task( key_t key, bool task_list_already_locked = false );
	void modify_task_priority( key_t key, int priority, bool task_list_already_locked );
public:
	void wait_task( key_t key );
	void modify_task_priority( key_t key, int priority );
	key_t claim_task( void* input, int priority );
	void  disclaim_task( key_t key );

private:

	auto_thread_pool atp;

	boost::mutex to_do_mutex;
	bool to_do;

	void set_to_do() { boost::lock_guard<boost::mutex> lock(to_do_mutex); to_do = true; }
	bool unset_to_do() {
		boost::lock_guard<boost::mutex> lock(to_do_mutex);
		bool p = to_do; to_do = false; return p; }

	boost::mutex daemon_switch_mutex;

	boost::mutex daemon_suicide_mutex;
	boost::shared_ptr< boost::lock_guard<boost::mutex> > no_suicide_lock;

	boost::shared_mutex processing_mutex;

	typedef boost::shared_lock<boost::shared_mutex> shared_lock_token_t;
	typedef std::map<shared_lock_token_t*, boost::shared_ptr<shared_lock_token_t> > token_list_t;
	boost::shared_mutex stop_mutex;
	token_list_t stop_request_list;
	boost::mutex stop_request_list_mutex;

public:
	typedef RSB::token_t token_t;
	token_t stop_processing();
	bool resume_processing( RSB::token_t authorized_token );
	void wait_for_resumed();
	void wait_for_stopped();

public:
	/// data pool
	class data_pool {
	private:
		// input & output
		boost::shared_mutex    vec_rw_mutex;

		std::vector<void*>     in;
		std::vector<void*>     out;
		std::vector<task_unit_wrapper> task;
		// key. the key pool growth only
		std::list<key_t> available_key;
		// functions
		boost::mutex data_pool_mutex;
	public:
		key_t require_key();
		void release_key( key_t key );
		void* out_ptr( key_t key );		// thread safe
		void* in_ptr( key_t key );		// thread safe
		void set_out_ptr( key_t key, void* p ); // thread safe, when operate for different key
		void set_in_ptr( key_t key, void* p ); // thread safe, when operate for different key
		task_unit& task_ref( key_t key );
	} dp;

	// thread & deamon
private:
	boost::shared_ptr< boost::thread > __daemon_thread;
	void __daemon();

	boost::mutex __daemon_loop_mutex;
	boost::shared_ptr< boost::lock_guard< boost::mutex> > __daemon_loop_lock;

	boost::mutex __daemon_loop_locked_off_mutex;
	boost::shared_ptr< boost::lock_guard< boost::mutex> >  __daemon_loop_locked_off_lock;

	boost::mutex __daemon_loop_locked_on_mutex;
	boost::shared_ptr< boost::lock_guard< boost::mutex> >  __daemon_loop_locked_on_lock;

	boost::mutex __unlock_daemon_loop_mutex;

	boost::mutex __try_unlock_mutex;
	boost::shared_ptr< boost::lock_guard<boost::mutex> > __daemon_no_unlock_region_lock;

	void __lock_daemon_loop();
	void __unlock_daemon_loop();

	size_t __worker_num;
	size_t __active_worker_num;

	mutable boost::shared_mutex __worker_num_mutex;
	mutable boost::mutex __active_worker_num_mutex;

	void __inc_active_worker_num();
	void __dec_active_worker_num();
	size_t __idle_worker_num();

private:
	void update_atp_worker_max_num(); 	// not thread safe only called by set_worker_num() and the constructor
public:
	size_t worker_num() const;
	void set_worker_num( size_t new_worker_num );	// return the previous worker num
	void daemon() { __daemon(); };
	void start_daemon();

public:
	static size_t parse_worker_num( size_t custom_worker_num );
};

///

size_t RSBI::parse_worker_num( size_t custom_worker_num ) {
	size_t new_worker_num = custom_worker_num;
	if ( new_worker_num == RSB::AUTO_WORKER_NUM() )
		new_worker_num = boost::thread::hardware_concurrency();
	if (!new_worker_num)
		new_worker_num = 1;
	return new_worker_num;
}

// ----------------------------------------------------------------------------------

// task operations

RSBI::__restaurant_scheduler_base_inner(
		const __type_concealed_handler& input_type_concealer,
		const __type_concealed_handler& output_type_concealer,
		const __single_worker_wrapper_base& prototype_worker_wrapper,
		size_t worker_num_ ) : __input_c(input_type_concealer.clone()),
				__output_c(output_type_concealer.clone()),
				__prototype_worker_wrapper( prototype_worker_wrapper.clone() ),
				to_do(false), __worker_num( worker_num_ ), __active_worker_num(0) {

	update_atp_worker_max_num();

	tasks[__restaurant_scheduler_base::TO_BE_DECALIMED_PRIORITY()];
	tasks[__restaurant_scheduler_base::IDLE_PRIORITY()];

	to_be_disclaimed_list_iter = tasks.find( __restaurant_scheduler_base::TO_BE_DECALIMED_PRIORITY() );
	idle_task_list_iter  = tasks.find( __restaurant_scheduler_base::IDLE_PRIORITY() );

}

RSBI::~__restaurant_scheduler_base_inner() {

	stop_processing();
	wait_for_stopped();

	// clear all the tasks

	{
		boost::lock_guard<boost::mutex> lock2( task_list_mutex );
		for ( std::map< int, std::list<key_t> >::iterator list_iter = tasks.begin();
				list_iter!=tasks.end(); ++list_iter ) {
			std::list<key_t>& l = ( list_iter->second );
			while( !l.empty() ) {
				key_t k = l.back();
				remove_task( k, true );
				// dp.release_key( k );	// not necessary
			}
		}

	}

}

void RSBI::set_task( RSBI::key_t key, void* input, int priority ) {
	{
		boost::lock_guard<boost::mutex> lock2( task_list_mutex );

		task_unit& t = dp.task_ref( key );

		// t.uncomplete_lock.reset( new boost::lock_guard
		//		<boost::mutex>( *t.uncomplete_mutex ) );

		boost::lock_guard<boost::mutex> lock1( *t.busy_mutex );

		dp.set_in_ptr( key, input );
		t.priority = priority;


		std::list<key_t>& this_list = tasks[priority];
		this_list.push_front( key );
		t.iter = this_list.begin();
	}

	__unlock_daemon_loop();

}

void RSBI::modify_task_priority( RSBI::key_t key, int priority, bool task_list_already_locked ) {

	boost::shared_ptr< boost::lock_guard<boost::mutex> > sptr_lock2;
	if (!task_list_already_locked)
		sptr_lock2.reset( new boost::lock_guard<boost::mutex> ( task_list_mutex ) );

	task_unit& t = dp.task_ref( key );

	boost::lock_guard<boost::mutex> lock1( *t.busy_mutex );

	if ( t.priority == priority || (t.priority >= __restaurant_scheduler_base::IDLE_PRIORITY() && t.priority>priority) )
		return; // do nothing

	std::map< int, std::list<key_t> >::iterator old_list_iter =
			tasks.find( t.priority );
	std::list<key_t>& new_list = tasks[ priority ];

	new_list.splice( new_list.begin(), old_list_iter->second, t.iter );
	t.priority = priority; t.iter = new_list.begin();

	if ( old_list_iter->first<__restaurant_scheduler_base::IDLE_PRIORITY() && old_list_iter->second.empty() )
		tasks.erase( old_list_iter );

}

void RSBI::modify_task_priority( RSBI::key_t key, int priority ) {
	modify_task_priority( key, priority, false );
}

void RSBI::remove_task( RSBI::key_t key, bool task_list_already_locked ) {

	boost::lock_guard<boost::mutex> lock0( task_removal_mutex );

	boost::shared_ptr< boost::lock_guard<boost::mutex> > sptr_lock2;
	if (!task_list_already_locked)
		sptr_lock2.reset( new boost::lock_guard<boost::mutex> ( task_list_mutex ) );

	task_unit& t = dp.task_ref( key );

	boost::lock_guard<boost::mutex> lock1( *t.busy_mutex );

	std::map< int, std::list<key_t> >::iterator this_list_iter =
			tasks.find( t.priority );
	this_list_iter->second.erase( t.iter );

	if ( t.priority < __restaurant_scheduler_base::IDLE_PRIORITY() &&
			this_list_iter->second.empty() ) {
		tasks.erase( this_list_iter );
	}

	if ( t.uncomplete_lock.get() ) {	// this is thread safe as the busy_mutex is obtained
		__input_c->delete_object( dp.in_ptr( key ) );
	} else {
		t.uncomplete_lock.reset( new boost::lock_guard
				<boost::mutex>( *t.uncomplete_mutex ) );
		__output_c->delete_object( dp.out_ptr( key ) );
	}

}

void RSBI::wait_task( RSBI::key_t key ) {
	task_unit& t = dp.task_ref( key );
	boost::lock_guard< boost::mutex > lock( *t.uncomplete_mutex );
}

RSBI::key_t RSBI::claim_task( void* input, int priority ) {
	key_t k = dp.require_key();
	set_task( k, input, priority );
	return k;
}

void  RSBI::disclaim_task( key_t key ) {
	modify_task_priority( key, __restaurant_scheduler_base::TO_BE_DECALIMED_PRIORITY() );
	__unlock_daemon_loop();
}

RSBI::token_t RSBI::stop_processing() {
	boost::lock_guard<boost::mutex> lock( stop_request_list_mutex );

	boost::shared_ptr<shared_lock_token_t> sptr_stop_lock( new shared_lock_token_t(stop_mutex) );

	if ( stop_request_list.empty() ) {
		boost::lock_guard<boost::mutex> lock( daemon_switch_mutex );
		no_suicide_lock.reset();
		__unlock_daemon_loop(); // give the daemon an opportunity to stop
	}

	stop_request_list.insert( std::make_pair( sptr_stop_lock.get(), sptr_stop_lock ) );

	return sptr_stop_lock.get();
}

bool RSBI::resume_processing( RSBI::token_t authorized_token ) {
	boost::lock_guard<boost::mutex> lock( stop_request_list_mutex );

	shared_lock_token_t* t = reinterpret_cast<shared_lock_token_t*>( authorized_token );
	token_list_t::iterator iter = stop_request_list.find( t );
	if ( iter == stop_request_list.end() )
		return false;
	stop_request_list.erase( iter );

	if ( stop_request_list.empty() ) {
		start_daemon();
		__unlock_daemon_loop();
	}

	return true;
}

void RSBI::wait_for_resumed() {
	boost::upgrade_lock<boost::shared_mutex> sharedLock( stop_mutex );
	boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock( sharedLock );
}

void RSBI::wait_for_stopped() {
	boost::upgrade_lock<boost::shared_mutex> sharedLock( processing_mutex );
	boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock( sharedLock );
	if (__daemon_thread.get())
		__daemon_thread->join();
}

class RSBI_daemon_wrapper {
	RSBI& __rsbi;
public:
	RSBI_daemon_wrapper( RSBI& rsbi ) : __rsbi(rsbi) {}
	void operator ()() {
		__rsbi.daemon();
	}
};

void RSBI::start_daemon() {

	wait_for_stopped();

	boost::lock_guard<boost::mutex> lock( daemon_switch_mutex );

	no_suicide_lock.reset( new boost::lock_guard<boost::mutex>( daemon_suicide_mutex ) );
	__daemon_no_unlock_region_lock.reset(
			new boost::lock_guard<boost::mutex>(__try_unlock_mutex) );

	__daemon_thread.reset( new boost::thread( RSBI_daemon_wrapper(*this) ) );
}

void RSBI::__lock_daemon_loop() {
	{
		// wait the loop state 2 to be ``unlocked''
		boost::lock_guard<boost::mutex> lock2( __daemon_loop_locked_on_mutex );
		// Lock the mutex for daemon loop
		__daemon_loop_lock.reset( new boost::lock_guard<boost::mutex>(__daemon_loop_mutex) );
	}
	// Set the loop state 2 to ``locked'
	__daemon_loop_locked_on_lock.reset( new boost::lock_guard
			<boost::mutex>( __daemon_loop_locked_on_mutex ) );
	// Set the loop state 1 to ``locked''
	__daemon_loop_locked_off_lock.reset();
}


void RSBI::__unlock_daemon_loop() {

	boost::lock_guard<boost::mutex> lock1( __unlock_daemon_loop_mutex );

	{
		{
			boost::unique_lock<boost::mutex> try_unlock_mutex(__try_unlock_mutex);
			set_to_do();
			// test whether the loop state 1 is ``locked''
			boost::unique_lock<boost::mutex> lock( __daemon_loop_locked_off_mutex, boost::try_to_lock );
			if (!lock) {
				return;
			}
		}

		// Unlock the mutex for deamon loop
		__daemon_loop_lock.reset();
	}

	// Set the loop state 1 to ``unlocked''
	__daemon_loop_locked_off_lock.reset( new boost::lock_guard
			<boost::mutex>( __daemon_loop_locked_off_mutex ) );
	// Set the loop state 2 to ``unlocked'
	__daemon_loop_locked_on_lock.reset();

}

void RSBI::__inc_active_worker_num() {
	boost::lock_guard< boost::mutex > lock(__active_worker_num_mutex);
	++__active_worker_num;
}
void RSBI::__dec_active_worker_num() {
	boost::lock_guard< boost::mutex > lock(__active_worker_num_mutex);
	--__active_worker_num;
}
size_t RSBI::__idle_worker_num() {
	boost::shared_lock< boost::shared_mutex > sharedLock(__worker_num_mutex);
	boost::lock_guard< boost::mutex > lock(__active_worker_num_mutex);
	return (__worker_num>__active_worker_num)?(__worker_num-__active_worker_num):0;
}

size_t RSBI::worker_num() const {
	boost::shared_lock< boost::shared_mutex > sharedLock(__worker_num_mutex);
	return __worker_num;
}

void RSBI::set_worker_num( size_t new_worker_num ) {
	bool need_unlock_loop;
	{
		boost::upgrade_lock< boost::shared_mutex > sharedLock(__worker_num_mutex);
		boost::upgrade_to_unique_lock< boost::shared_mutex > uniqueLock(sharedLock);
		need_unlock_loop = ( __worker_num<new_worker_num );
		__worker_num = new_worker_num;
		update_atp_worker_max_num();
	}
	if ( need_unlock_loop )
		__unlock_daemon_loop();
}

void RSBI::update_atp_worker_max_num() {
	size_t apt_worker_max_num = std::max( size_t(3), std::max( __worker_num*2u,
			std::min( __worker_num*3u, size_t(boost::thread::hardware_concurrency())*3u )) );
	atp.set_max_allowed_worker_num( apt_worker_max_num );
}


class RSBI_worker {
private:
	RSBI& __h;
	RSBI::key_t __key;
	boost::shared_ptr< boost::shared_lock<boost::shared_mutex> > __processing_lock;
public:
	RSBI_worker( RSBI& h, key_t key ) :
		__h(h), __key(key),
		__processing_lock( new boost::shared_lock<boost::shared_mutex>(__h.processing_mutex) ) {}
	void operator()() {
		RSBI::task_unit& t = __h.dp.task_ref(__key);
		{
			boost::lock_guard<boost::mutex> ( *(t.busy_mutex) );
			void * bp;
			void * a = __h.dp.in_ptr(__key);
			{
				boost::shared_ptr<__single_worker_wrapper_base> fun( __h.__prototype_worker_wrapper->clone() );
				bp = (*fun)(a);
			}
			__h.dp.set_out_ptr(__key,bp);
			__h.__input_c->delete_object(a);
			t.uncomplete_lock.reset();
			__h.__dec_active_worker_num();
		}
		__h.__unlock_daemon_loop();
	}
};



void RSBI::__daemon() {
	boost::shared_lock<boost::shared_mutex> processing_lock(processing_mutex);
	bool is_on = true;

	while (is_on) {
		{
			// NO unlock region CONTINUE
			__lock_daemon_loop();	// lock this iteration

			__daemon_no_unlock_region_lock.reset();
			// NO unlock region END

			// wait for some event to unlock the loop
			boost::lock_guard<boost::mutex> lock1(__daemon_loop_mutex);

			// if it continues it means something need to be done
			// (if more than one things happen concurrently, they might be done in a single iteration,
			// so sometimes there may be nothing to be done in this iteration)

			// stop signal ??

			while( unset_to_do() ) {

				__daemon_no_unlock_region_lock.reset();

				boost::unique_lock<boost::mutex> suicide_lock( daemon_suicide_mutex, boost::try_to_lock );
				if (suicide_lock) {
					is_on = false;
					// stop and leave the loop lock open
					break;
				}

				// PROCESSING
				while ( __idle_worker_num() ) {
					boost::lock_guard<boost::mutex> lock0( task_removal_mutex );

					// check whether there is something need me to do
					key_t pending_key = __restaurant_scheduler_base::NULL_KEY();
					{
						boost::lock_guard<boost::mutex> lock2( task_list_mutex );

						for ( std::map< int, std::list<key_t> >::iterator list_iter = tasks.begin();
								list_iter!=idle_task_list_iter; ++list_iter ) {
							if ( list_iter->second.empty() )	// this condition should be USELESS
								continue;
							pending_key = list_iter->second.back();
							break;
						}
						if ( pending_key == __restaurant_scheduler_base::NULL_KEY() )
							break;

						// move task from pending list to idle list
						modify_task_priority( pending_key, __restaurant_scheduler_base::IDLE_PRIORITY(), true );

					}

					// do the task
					{
						RSBI_worker riw( *this, pending_key );
						// boost::thread t(riw);
						// t.detach();
						atp.run( riw );
						__inc_active_worker_num();
					}

				}

				// handle disclaimed list

				{
					boost::lock_guard<boost::mutex> lock0( task_list_mutex );
					std::list<key_t>& to_be_disclaimed_list = to_be_disclaimed_list_iter->second;
					while( !to_be_disclaimed_list.empty() ) {
						key_t k = to_be_disclaimed_list.back();
						remove_task( k, true );
						dp.release_key( k );
					}
				}

				__daemon_no_unlock_region_lock.reset(
						new boost::lock_guard<boost::mutex>(__try_unlock_mutex) );
			}

			// NO unlock region START from just before last '}'


		}

	}

}


// data pool operators

RSBI::key_t RSBI::data_pool::require_key() {

	boost::lock_guard<boost::mutex> lock1( data_pool_mutex );

	if ( available_key.empty() ) {
		boost::upgrade_lock<boost::shared_mutex> sharedLock( vec_rw_mutex );
		boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(sharedLock);

		/* enlarge data pool */
		static const size_t MINIMUM_KEY_POOL_SIZE = 16;
		size_t cur_size = in.size();
		if (cur_size<MINIMUM_KEY_POOL_SIZE )
			cur_size = MINIMUM_KEY_POOL_SIZE;
		else
			cur_size *= 2u;

		for ( key_t k = in.size(); k<cur_size; ++k )
			available_key.push_front( k );

		__resize_vector_with_default_constructor( in,   cur_size );
		__resize_vector_with_default_constructor( out,  cur_size );
		__resize_vector_with_default_constructor( task, cur_size );
	}

	key_t k = available_key.back();
	available_key.pop_back();

	return k;

}

void RSBI::data_pool::release_key( RSBI::key_t key ) {
	boost::lock_guard<boost::mutex> lock( data_pool_mutex );
	available_key.push_back( key );
}

inline void RSBI::data_pool::set_in_ptr( RSBI::key_t key, void* p ) {
	boost::shared_lock<boost::shared_mutex> sharedLock( vec_rw_mutex );
	in[key] = p;
}

inline void RSBI::data_pool::set_out_ptr( RSBI::key_t key, void* p ) {
	boost::shared_lock<boost::shared_mutex> sharedLock( vec_rw_mutex );
	out[key] = p;
}

inline void* RSBI::data_pool::in_ptr( RSBI::key_t key ) {
	boost::shared_lock<boost::shared_mutex> sharedLock( vec_rw_mutex );
	return in[key];
}


inline void* RSBI::data_pool::out_ptr( RSBI::key_t key ) {
	boost::shared_lock<boost::shared_mutex> sharedLock( vec_rw_mutex );
	return out[key];
}

inline RSBI::task_unit& RSBI::data_pool::task_ref( key_t key ) {
	boost::shared_lock<boost::shared_mutex> sharedLock( vec_rw_mutex );
	return *(task[key].p);
}


// -------------------------------------------------------------------------------

RSB::__restaurant_scheduler_base(
		const __type_concealed_handler& input_type_concealer,
		const __type_concealed_handler& output_type_concealer,
		const __single_worker_wrapper_base& prototype_worker_wrapper,
		size_t worker_num ) : __h(
				new __restaurant_scheduler_base_inner(
						input_type_concealer,
						output_type_concealer,
						prototype_worker_wrapper,
						RSBI::parse_worker_num(worker_num) ) ) {
	__h->start_daemon();
}
RSB::~__restaurant_scheduler_base() {
	delete __h;
}

RSB::key_t  RSB::_claim_key( void* in_ptr, int priority ) {
	return __h->claim_task( in_ptr, priority );
}

void  RSB::_change_priority( key_t key, int priority ) {
	__h->modify_task_priority( key, priority );
}

void  RSB::_disclaim_key(  key_t key ) {
	__h->disclaim_task( key );
}

void* RSB::_in_ptr( key_t key ) {
	return __h->dp.in_ptr( key );
}

void* RSB::_out_ptr( key_t key ) {
	return __h->dp.out_ptr( key );
}

void  RSB::_wait_key( key_t key ) {
	__h->wait_task( key );
}

size_t RSB::worker_num() const {
	return __h->worker_num();
}

void RSB::set_worker_num( size_t new_worker_num ) {
	__h->set_worker_num( RSBI::parse_worker_num(new_worker_num) );
}

RSB::token_t RSB::stop_processing() {
	return __h->stop_processing();
}

bool RSB::resume_processing( RSB::token_t authorized_token ) {
	return __h->resume_processing( authorized_token );
}

void RSB::wait_for_resumed() {
	__h->wait_for_resumed();
}

void RSB::wait_for_stopped() {
	__h->wait_for_stopped();
}


}
