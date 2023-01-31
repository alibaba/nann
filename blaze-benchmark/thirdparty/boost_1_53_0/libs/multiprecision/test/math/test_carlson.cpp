///////////////////////////////////////////////////////////////
//  Copyright 2011 John Maddock. Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_

#ifdef _MSC_VER
#  define _SCL_SECURE_NO_WARNINGS
#endif

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#if !defined(TEST_MPF_50) && !defined(TEST_BACKEND) && !defined(TEST_CPP_DEC_FLOAT) && !defined(TEST_MPFR_50)
#  define TEST_MPF_50
#  define TEST_MPFR_50
#  define TEST_CPP_DEC_FLOAT

#ifdef _MSC_VER
#pragma message("CAUTION!!: No backend type specified so testing everything.... this will take some time!!")
#endif
#ifdef __GNUC__
#pragma warning "CAUTION!!: No backend type specified so testing everything.... this will take some time!!"
#endif

#endif

#if defined(TEST_MPF_50)
#include <boost/multiprecision/gmp.hpp>
#endif
#if defined(TEST_MPFR_50)
#include <boost/multiprecision/mpfr.hpp>
#endif
#ifdef TEST_BACKEND
#include <boost/multiprecision/concepts/mp_number_archetypes.hpp>
#endif
#ifdef TEST_CPP_DEC_FLOAT
#include <boost/multiprecision/cpp_dec_float.hpp>
#endif

#include "table_type.hpp"
#define TEST_UDT

#include <boost/math/special_functions/ellint_rc.hpp>
#include <boost/math/special_functions/ellint_rj.hpp>
#include <boost/math/special_functions/ellint_rd.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>
#include "libs/math/test/test_carlson.hpp"

void expected_results()
{
   //
   // Define the max and mean errors expected for
   // various compilers and platforms.
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                          // test type(s)
      ".*RJ.*",                      // test data group
      ".*", 2700, 250);              // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                          // test type(s)
      ".*",                          // test data group
      ".*", 40, 20);                 // test function
   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", " 
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}


int test_main(int, char* [])
{
   using namespace boost::multiprecision;
   expected_results();
   //
   // Test at:
   // 18 decimal digits: tests 80-bit long double approximations
   // 30 decimal digits: tests 128-bit long double approximations
   // 35 decimal digits: tests arbitrary precision code
   //
#ifdef TEST_MPF_50
   // We have accuracy issues with gmp_float<18> - our argument reduction in std::sin isn't accurate enough:
   test_spots(number<gmp_float<18> >(), "number<gmp_float<18> >");
   test_spots(number<gmp_float<30> >(), "number<gmp_float<30> >");
   test_spots(number<gmp_float<35> >(), "number<gmp_float<35> >");
   // there should be at least one test with expression templates off:
   test_spots(number<gmp_float<35>, et_off>(), "number<gmp_float<35>, et_off>");
#endif
#ifdef TEST_MPFR_50
   test_spots(number<mpfr_float_backend<18> >(), "number<mpfr_float_backend<18> >");
   test_spots(number<mpfr_float_backend<30> >(), "number<mpfr_float_backend<30> >");
   test_spots(number<mpfr_float_backend<35> >(), "number<mpfr_float_backend<35> >");
#endif
#ifdef TEST_CPP_DEC_FLOAT
   test_spots(number<cpp_dec_float<18> >(), "number<cpp_dec_float<18> >");
   test_spots(number<cpp_dec_float<30> >(), "number<cpp_dec_float<30> >");
   test_spots(number<cpp_dec_float<35, long long, std::allocator<void> > >(), "number<cpp_dec_float<35, long long, std::allocator<void> > >");
#endif
   return 0;
}



