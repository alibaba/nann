//  Copyright (C) 2011 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/lockfree/spsc_queue.hpp>

#include <boost/thread.hpp>

#define BOOST_TEST_MAIN
#ifdef BOOST_LOCKFREE_INCLUDE_TESTS
#include <boost/test/included/unit_test.hpp>
#else
#include <boost/test/unit_test.hpp>
#endif

#include <iostream>
#include <memory>

#include "test_helpers.hpp"
#include "test_common.hpp"

using namespace boost;
using namespace boost::lockfree;
using namespace std;

BOOST_AUTO_TEST_CASE( simple_spsc_queue_test )
{
    spsc_queue<int, capacity<64> > f;

    BOOST_REQUIRE(f.empty());
    f.push(1);
    f.push(2);

    int i1(0), i2(0);

    BOOST_REQUIRE(f.pop(i1));
    BOOST_REQUIRE_EQUAL(i1, 1);

    BOOST_REQUIRE(f.pop(i2));
    BOOST_REQUIRE_EQUAL(i2, 2);
    BOOST_REQUIRE(f.empty());
}

BOOST_AUTO_TEST_CASE( simple_spsc_queue_test_compile_time_size )
{
    spsc_queue<int> f(64);

    BOOST_REQUIRE(f.empty());
    f.push(1);
    f.push(2);

    int i1(0), i2(0);

    BOOST_REQUIRE(f.pop(i1));
    BOOST_REQUIRE_EQUAL(i1, 1);

    BOOST_REQUIRE(f.pop(i2));
    BOOST_REQUIRE_EQUAL(i2, 2);
    BOOST_REQUIRE(f.empty());
}

BOOST_AUTO_TEST_CASE( ranged_push_test )
{
    spsc_queue<int> stk(64);

    int data[2] = {1, 2};

    BOOST_REQUIRE_EQUAL(stk.push(data, data + 2), data + 2);

    int out;
    BOOST_REQUIRE(stk.pop(out)); BOOST_REQUIRE_EQUAL(out, 1);
    BOOST_REQUIRE(stk.pop(out)); BOOST_REQUIRE_EQUAL(out, 2);
    BOOST_REQUIRE(!stk.pop(out));
}


enum {
    pointer_and_size,
    reference_to_array,
    iterator_pair,
    output_iterator_
};


template <int EnqueueMode>
void spsc_queue_buffer_push_return_value(void)
{
    const size_t xqueue_size = 64;
    const size_t buffer_size = 100;
    spsc_queue<int, capacity<100> > rb;

    int data[xqueue_size];
    for (size_t i = 0; i != xqueue_size; ++i)
        data[i] = i*2;

    switch (EnqueueMode) {
    case pointer_and_size:
        BOOST_REQUIRE_EQUAL(rb.push(data, xqueue_size), xqueue_size);
        break;

    case reference_to_array:
        BOOST_REQUIRE_EQUAL(rb.push(data), xqueue_size);
        break;

    case iterator_pair:
        BOOST_REQUIRE_EQUAL(rb.push(data, data + xqueue_size), data + xqueue_size);
        break;

    default:
        assert(false);
    }

    switch (EnqueueMode) {
    case pointer_and_size:
        BOOST_REQUIRE_EQUAL(rb.push(data, xqueue_size), buffer_size - xqueue_size - 1);
        break;

    case reference_to_array:
        BOOST_REQUIRE_EQUAL(rb.push(data), buffer_size - xqueue_size - 1);
        break;

    case iterator_pair:
        BOOST_REQUIRE_EQUAL(rb.push(data, data + xqueue_size), data + buffer_size - xqueue_size - 1);
        break;

    default:
        assert(false);
    }
}

BOOST_AUTO_TEST_CASE( spsc_queue_buffer_push_return_value_test )
{
    spsc_queue_buffer_push_return_value<pointer_and_size>();
    spsc_queue_buffer_push_return_value<reference_to_array>();
    spsc_queue_buffer_push_return_value<iterator_pair>();
}

template <int EnqueueMode,
          int ElementCount,
          int BufferSize,
          int NumberOfIterations
         >
void spsc_queue_buffer_push(void)
{
    const size_t xqueue_size = ElementCount;
    spsc_queue<int, capacity<BufferSize> > rb;

    int data[xqueue_size];
    for (size_t i = 0; i != xqueue_size; ++i)
        data[i] = i*2;

    std::vector<int> vdata(data, data + xqueue_size);

    for (int i = 0; i != NumberOfIterations; ++i) {
        BOOST_REQUIRE(rb.empty());
        switch (EnqueueMode) {
        case pointer_and_size:
            BOOST_REQUIRE_EQUAL(rb.push(data, xqueue_size), xqueue_size);
            break;

        case reference_to_array:
            BOOST_REQUIRE_EQUAL(rb.push(data), xqueue_size);
            break;

        case iterator_pair:
            BOOST_REQUIRE_EQUAL(rb.push(data, data + xqueue_size), data + xqueue_size);
            break;

        default:
            assert(false);
        }

        int out[xqueue_size];
        BOOST_REQUIRE_EQUAL(rb.pop(out, xqueue_size), xqueue_size);
        for (size_t i = 0; i != xqueue_size; ++i)
            BOOST_REQUIRE_EQUAL(data[i], out[i]);
    }
}

BOOST_AUTO_TEST_CASE( spsc_queue_buffer_push_test )
{
    spsc_queue_buffer_push<pointer_and_size, 7, 16, 64>();
    spsc_queue_buffer_push<reference_to_array, 7, 16, 64>();
    spsc_queue_buffer_push<iterator_pair, 7, 16, 64>();
}

template <int EnqueueMode,
          int ElementCount,
          int BufferSize,
          int NumberOfIterations
         >
void spsc_queue_buffer_pop(void)
{
    const size_t xqueue_size = ElementCount;
    spsc_queue<int, capacity<BufferSize> > rb;

    int data[xqueue_size];
    for (size_t i = 0; i != xqueue_size; ++i)
        data[i] = i*2;

    std::vector<int> vdata(data, data + xqueue_size);

    for (int i = 0; i != NumberOfIterations; ++i) {
        BOOST_REQUIRE(rb.empty());
        BOOST_REQUIRE_EQUAL(rb.push(data), xqueue_size);

        int out[xqueue_size];
        vector<int> vout;

        switch (EnqueueMode) {
        case pointer_and_size:
            BOOST_REQUIRE_EQUAL(rb.pop(out, xqueue_size), xqueue_size);
            break;

        case reference_to_array:
            BOOST_REQUIRE_EQUAL(rb.pop(out), xqueue_size);
            break;

        case output_iterator_:
            BOOST_REQUIRE_EQUAL(rb.pop(std::back_inserter(vout)), xqueue_size);
            break;

        default:
            assert(false);
        }

        if (EnqueueMode == output_iterator_) {
            BOOST_REQUIRE_EQUAL(vout.size(), xqueue_size);
            for (size_t i = 0; i != xqueue_size; ++i)
                BOOST_REQUIRE_EQUAL(data[i], vout[i]);
        } else {
            for (size_t i = 0; i != xqueue_size; ++i)
                BOOST_REQUIRE_EQUAL(data[i], out[i]);
        }
    }
}

BOOST_AUTO_TEST_CASE( spsc_queue_buffer_pop_test )
{
    spsc_queue_buffer_pop<pointer_and_size, 7, 16, 64>();
    spsc_queue_buffer_pop<reference_to_array, 7, 16, 64>();
    spsc_queue_buffer_pop<output_iterator_, 7, 16, 64>();
}


#ifndef BOOST_LOCKFREE_STRESS_TEST
static const boost::uint32_t nodes_per_thread = 100000;
#else
static const boost::uint32_t nodes_per_thread = 100000000;
#endif

struct spsc_queue_tester
{
    spsc_queue<int, capacity<128> > sf;

    boost::lockfree::detail::atomic<long> spsc_queue_cnt, received_nodes;

    static_hashed_set<int, 1<<16 > working_set;

    spsc_queue_tester(void):
        spsc_queue_cnt(0), received_nodes(0)
    {}

    void add(void)
    {
        for (boost::uint32_t i = 0; i != nodes_per_thread; ++i) {
            int id = generate_id<int>();
            working_set.insert(id);

            while (sf.push(id) == false)
            {}

            ++spsc_queue_cnt;
        }
        running = false;
    }

    bool get_element(void)
    {
        int data;
        bool success = sf.pop(data);

        if (success) {
            ++received_nodes;
            --spsc_queue_cnt;
            bool erased = working_set.erase(data);
            assert(erased);
            return true;
        } else
            return false;
    }

    boost::lockfree::detail::atomic<bool> running;

    void get(void)
    {
        for(;;) {
            bool success = get_element();
            if (!running && !success)
                break;
        }

        while ( get_element() );
    }

    void run(void)
    {
        running = true;

        BOOST_REQUIRE(sf.empty());

        thread reader(boost::bind(&spsc_queue_tester::get, this));
        thread writer(boost::bind(&spsc_queue_tester::add, this));
        cout << "reader and writer threads created" << endl;

        writer.join();
        cout << "writer threads joined. waiting for readers to finish" << endl;

        reader.join();

        BOOST_REQUIRE_EQUAL(received_nodes, nodes_per_thread);
        BOOST_REQUIRE_EQUAL(spsc_queue_cnt, 0);
        BOOST_REQUIRE(sf.empty());
        BOOST_REQUIRE(working_set.count_nodes() == 0);
    }
};

BOOST_AUTO_TEST_CASE( spsc_queue_test_caching )
{
    boost::shared_ptr<spsc_queue_tester> test1(new spsc_queue_tester);
    test1->run();
}

struct spsc_queue_tester_buffering
{
    spsc_queue<int, capacity<128> > sf;

    boost::lockfree::detail::atomic<long> spsc_queue_cnt;

    static_hashed_set<int, 1<<16 > working_set;
    boost::lockfree::detail::atomic<long> received_nodes;

    spsc_queue_tester_buffering(void):
        spsc_queue_cnt(0), received_nodes(0)
    {}

    static const size_t buf_size = 5;

    void add(void)
    {
        boost::array<int, buf_size> input_buffer;
        for (boost::uint32_t i = 0; i != nodes_per_thread; i+=buf_size) {
            for (size_t i = 0; i != buf_size; ++i) {
                int id = generate_id<int>();
                working_set.insert(id);
                input_buffer[i] = id;
            }

            size_t pushed = 0;

            do {
                pushed += sf.push(input_buffer.c_array() + pushed,
                                  input_buffer.size()    - pushed);
            } while (pushed != buf_size);

            spsc_queue_cnt+=buf_size;
        }
        running = false;
    }

    bool get_elements(void)
    {
        boost::array<int, buf_size> output_buffer;

        size_t popd = sf.pop(output_buffer.c_array(), output_buffer.size());

        if (popd) {
            received_nodes += popd;
            spsc_queue_cnt -= popd;

            for (size_t i = 0; i != popd; ++i) {
                bool erased = working_set.erase(output_buffer[i]);
                assert(erased);
            }

            return true;
        } else
            return false;
    }

    boost::lockfree::detail::atomic<bool> running;

    void get(void)
    {
        for(;;) {
            bool success = get_elements();
            if (!running && !success)
                break;
        }

        while ( get_elements() );
    }

    void run(void)
    {
        running = true;

        thread reader(boost::bind(&spsc_queue_tester_buffering::get, this));
        thread writer(boost::bind(&spsc_queue_tester_buffering::add, this));
        cout << "reader and writer threads created" << endl;

        writer.join();
        cout << "writer threads joined. waiting for readers to finish" << endl;

        reader.join();

        BOOST_REQUIRE_EQUAL(received_nodes, nodes_per_thread);
        BOOST_REQUIRE_EQUAL(spsc_queue_cnt, 0);
        BOOST_REQUIRE(sf.empty());
        BOOST_REQUIRE(working_set.count_nodes() == 0);
    }
};


BOOST_AUTO_TEST_CASE( spsc_queue_test_buffering )
{
    boost::shared_ptr<spsc_queue_tester_buffering> test1(new spsc_queue_tester_buffering);
    test1->run();
}
