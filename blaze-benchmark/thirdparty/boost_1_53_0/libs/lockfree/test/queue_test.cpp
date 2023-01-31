//  Copyright (C) 2011 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/lockfree/queue.hpp>
#include <boost/thread.hpp>

#define BOOST_TEST_MAIN
#ifdef BOOST_LOCKFREE_INCLUDE_TESTS
#include <boost/test/included/unit_test.hpp>
#else
#include <boost/test/unit_test.hpp>
#endif

#include <memory>

using namespace boost;
using namespace boost::lockfree;
using namespace std;

BOOST_AUTO_TEST_CASE( simple_queue_test )
{
    queue<int> f(64);

    BOOST_WARN(f.is_lock_free());

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

BOOST_AUTO_TEST_CASE( simple_queue_test_capacity )
{
    queue<int, capacity<64> > f;

    BOOST_WARN(f.is_lock_free());

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


BOOST_AUTO_TEST_CASE( unsafe_queue_test )
{
    queue<int> f(64);

    BOOST_WARN(f.is_lock_free());
    BOOST_REQUIRE(f.empty());

    int i1(0), i2(0);

    f.unsynchronized_push(1);
    f.unsynchronized_push(2);

    BOOST_REQUIRE(f.unsynchronized_pop(i1));
    BOOST_REQUIRE_EQUAL(i1, 1);

    BOOST_REQUIRE(f.unsynchronized_pop(i2));
    BOOST_REQUIRE_EQUAL(i2, 2);
    BOOST_REQUIRE(f.empty());
}


BOOST_AUTO_TEST_CASE( queue_convert_pop_test )
{
    queue<int*> f(128);
    BOOST_REQUIRE(f.empty());
    f.push(new int(1));
    f.push(new int(2));
    f.push(new int(3));
    f.push(new int(4));

    {
        int * i1;

        BOOST_REQUIRE(f.pop(i1));
        BOOST_REQUIRE_EQUAL(*i1, 1);
        delete i1;
    }


    {
        boost::shared_ptr<int> i2;
        BOOST_REQUIRE(f.pop(i2));
        BOOST_REQUIRE_EQUAL(*i2, 2);
    }

    {
        auto_ptr<int> i3;
        BOOST_REQUIRE(f.pop(i3));

        BOOST_REQUIRE_EQUAL(*i3, 3);
    }

    {
        boost::shared_ptr<int> i4;
        BOOST_REQUIRE(f.pop(i4));

        BOOST_REQUIRE_EQUAL(*i4, 4);
    }


    BOOST_REQUIRE(f.empty());
}
