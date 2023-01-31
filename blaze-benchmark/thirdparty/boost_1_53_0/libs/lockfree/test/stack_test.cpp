//  Copyright (C) 2011 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <boost/thread.hpp>
#include <boost/lockfree/stack.hpp>

#define BOOST_TEST_MAIN
#ifdef BOOST_LOCKFREE_INCLUDE_TESTS
#include <boost/test/included/unit_test.hpp>
#else
#include <boost/test/unit_test.hpp>
#endif

BOOST_AUTO_TEST_CASE( simple_stack_test )
{
    boost::lockfree::stack<long> stk(128);

    stk.push(1);
    stk.push(2);
    long out;
    BOOST_REQUIRE(stk.pop(out)); BOOST_REQUIRE_EQUAL(out, 2);
    BOOST_REQUIRE(stk.pop(out)); BOOST_REQUIRE_EQUAL(out, 1);
    BOOST_REQUIRE(!stk.pop(out));
}

BOOST_AUTO_TEST_CASE( unsafe_stack_test )
{
    boost::lockfree::stack<long> stk(128);

    stk.unsynchronized_push(1);
    stk.unsynchronized_push(2);
    long out;
    BOOST_REQUIRE(stk.unsynchronized_pop(out)); BOOST_REQUIRE_EQUAL(out, 2);
    BOOST_REQUIRE(stk.unsynchronized_pop(out)); BOOST_REQUIRE_EQUAL(out, 1);
    BOOST_REQUIRE(!stk.unsynchronized_pop(out));
}

BOOST_AUTO_TEST_CASE( ranged_push_test )
{
    boost::lockfree::stack<long> stk(128);

    long data[2] = {1, 2};

    BOOST_REQUIRE_EQUAL(stk.push(data, data + 2), data + 2);

    long out;
    BOOST_REQUIRE(stk.unsynchronized_pop(out)); BOOST_REQUIRE_EQUAL(out, 2);
    BOOST_REQUIRE(stk.unsynchronized_pop(out)); BOOST_REQUIRE_EQUAL(out, 1);
    BOOST_REQUIRE(!stk.unsynchronized_pop(out));
}

BOOST_AUTO_TEST_CASE( ranged_unsynchronized_push_test )
{
    boost::lockfree::stack<long> stk(128);

    long data[2] = {1, 2};

    BOOST_REQUIRE_EQUAL(stk.unsynchronized_push(data, data + 2), data + 2);

    long out;
    BOOST_REQUIRE(stk.unsynchronized_pop(out)); BOOST_REQUIRE_EQUAL(out, 2);
    BOOST_REQUIRE(stk.unsynchronized_pop(out)); BOOST_REQUIRE_EQUAL(out, 1);
    BOOST_REQUIRE(!stk.unsynchronized_pop(out));
}

BOOST_AUTO_TEST_CASE( fixed_size_stack_test )
{
    boost::lockfree::stack<long, boost::lockfree::capacity<128> > stk;

    stk.push(1);
    stk.push(2);
    long out;
    BOOST_REQUIRE(stk.pop(out)); BOOST_REQUIRE_EQUAL(out, 2);
    BOOST_REQUIRE(stk.pop(out)); BOOST_REQUIRE_EQUAL(out, 1);
    BOOST_REQUIRE(!stk.pop(out));
    BOOST_REQUIRE(stk.empty());
}

BOOST_AUTO_TEST_CASE( fixed_size_stack_test_exhausted )
{
    boost::lockfree::stack<long, boost::lockfree::capacity<2> > stk;

    stk.push(1);
    stk.push(2);
    BOOST_REQUIRE(!stk.push(3));
    long out;
    BOOST_REQUIRE(stk.pop(out)); BOOST_REQUIRE_EQUAL(out, 2);
    BOOST_REQUIRE(stk.pop(out)); BOOST_REQUIRE_EQUAL(out, 1);
    BOOST_REQUIRE(!stk.pop(out));
    BOOST_REQUIRE(stk.empty());
}

BOOST_AUTO_TEST_CASE( bounded_stack_test_exhausted )
{
    boost::lockfree::stack<long> stk(2);

    stk.bounded_push(1);
    stk.bounded_push(2);
    BOOST_REQUIRE(!stk.bounded_push(3));
    long out;
    BOOST_REQUIRE(stk.pop(out)); BOOST_REQUIRE_EQUAL(out, 2);
    BOOST_REQUIRE(stk.pop(out)); BOOST_REQUIRE_EQUAL(out, 1);
    BOOST_REQUIRE(!stk.pop(out));
    BOOST_REQUIRE(stk.empty());
}
