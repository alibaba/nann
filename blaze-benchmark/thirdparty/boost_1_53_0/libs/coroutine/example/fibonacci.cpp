
//          Copyright Oliver Kowalke 2009.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cstdlib>
#include <iostream>

#include <boost/assert.hpp>
#include <boost/range.hpp>
#include <boost/coroutine/all.hpp>

typedef boost::coroutines::coroutine< int() >         coro_t;
typedef boost::range_iterator< coro_t >::type   iterator_t;

void fibonacci( coro_t::caller_type & c)
{
    int first = 1, second = 1;
    while ( true)
    {
        int third = first + second;
        first = second;
        second = third;
        c( third);     
    }
}

int main()
{
    coro_t c( fibonacci);
    iterator_t it( boost::begin( c) );
    BOOST_ASSERT( boost::end( c) != it);
    for ( int i = 0; i < 10; ++i)
    {
        std::cout << * it <<  " ";
        ++it;
    }

    std::cout << "\nDone" << std::endl;

    return EXIT_SUCCESS;
}
