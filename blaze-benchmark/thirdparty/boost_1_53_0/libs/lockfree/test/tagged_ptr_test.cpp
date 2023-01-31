//  Copyright (C) 2011 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/lockfree/detail/tagged_ptr.hpp>

#define BOOST_TEST_MAIN
#ifdef BOOST_LOCKFREE_INCLUDE_TESTS
#include <boost/test/included/unit_test.hpp>
#else
#include <boost/test/unit_test.hpp>
#endif

BOOST_AUTO_TEST_CASE( tagged_ptr_test )
{
    using namespace boost::lockfree::detail;
    int a(1), b(2);

    {
        tagged_ptr<int> i (&a, 0);
        tagged_ptr<int> j (&b, 1);

        i = j;

        BOOST_REQUIRE_EQUAL(i.get_ptr(), &b);
        BOOST_REQUIRE_EQUAL(i.get_tag(), 1);
    }

    {
        tagged_ptr<int> i (&a, 0);
        tagged_ptr<int> j (i);

        BOOST_REQUIRE_EQUAL(i.get_ptr(), j.get_ptr());
        BOOST_REQUIRE_EQUAL(i.get_tag(), j.get_tag());
    }

}
