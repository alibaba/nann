/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * test_timer.cpp
 *
 *  Created on: Jun 25, 2014
 *      Author: vpoliboy
 */

#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/thread.hpp>
#include <boost/foreach.hpp>
#include "cppmetrics/core/timer.h"

namespace cppmetrics {
namespace core {

TEST(timer, initialTest) {
    Timer timer;
    ASSERT_EQ((boost::uint64_t )0, timer.getCount());
    ASSERT_NEAR(0.0, timer.getMeanRate(), 0.000001);
    ASSERT_NEAR(0.0, timer.getOneMinuteRate(), 0.000001);
    ASSERT_NEAR(0.0, timer.getFiveMinuteRate(), 0.000001);
    ASSERT_NEAR(0.0, timer.getFifteenMinuteRate(), 0.000001);

    timer.update(boost::chrono::seconds(1));
    ASSERT_EQ((boost::uint64_t )1, timer.getCount());
}

TEST(timer, timerContextTest) {
    Timer timer;
    boost::mt11213b rng;
    for (size_t i = 0; i < 100; ++i) {
        boost::random::uniform_int_distribution<> uniform(10, 30);
        size_t sleep_time = uniform(rng);
        TimerContextPtr time_context(timer.timerContextPtr());
        boost::this_thread::sleep(boost::posix_time::milliseconds(sleep_time));
    }
    ASSERT_EQ((boost::uint64_t )100, timer.getCount());
    SnapshotPtr snapshot = timer.getSnapshot();
    // On jenkins builds, when there is lot of load, the duration of the sleep
    // in the timerContextTest takes more than the 20 ns. This is to eliminate
    // the periodic test failures during CI.

    std::cout << "             count = " << timer.getCount() << std::endl;
    std::cout << "         mean rate = " << (timer.getMeanRate())
            << " calls per 1 sec" << std::endl;
    std::cout << "     1-minute rate = " << (timer.getOneMinuteRate())
            << " calls per 1 sec" << std::endl;
    std::cout << "     5-minute rate = " << (timer.getFiveMinuteRate())
            << " calls per 1 sec" << std::endl;
    std::cout << "    15-minute rate = " << (timer.getFifteenMinuteRate())
            << " calls per 1 sec" << std::endl;
    std::cout << "               min = " << (snapshot->getMin()) << " ns "
            << std::endl;
    std::cout << "               max = " << (snapshot->getMax()) << " ns "
            << std::endl;
    std::cout << "              mean = " << (snapshot->getMean()) << " ns "
            << std::endl;
    std::cout << "            stddev = " << (snapshot->getStdDev()) << " ns "
            << std::endl;
    std::cout << "            median = " << (snapshot->getMedian()) << " ns "
            << std::endl;
    std::cout << "              75% <= " << (snapshot->get75thPercentile())
            << " ns " << std::endl;
    std::cout << "              95% <= " << (snapshot->get95thPercentile())
            << " ns " << std::endl;
    std::cout << "              98% <= " << (snapshot->get98thPercentile())
            << " ns " << std::endl;
    std::cout << "              99% <= " << (snapshot->get99thPercentile())
            << " ns " << std::endl;
    std::cout << "            99.9% <= " << (snapshot->get999thPercentile())
            << " ns " << std::endl;
    ASSERT_LE(25, static_cast<int>(timer.getMeanRate()));
    ASSERT_GE(55, static_cast<int>(timer.getMeanRate()));
    ASSERT_LE(
            boost::chrono::duration_cast<boost::chrono::nanoseconds>(
                    boost::chrono::milliseconds(8)).count(),
            static_cast<int>(snapshot->getMin()));
    ASSERT_GE(
            boost::chrono::duration_cast<boost::chrono::nanoseconds>(
                    boost::chrono::milliseconds(45)).count(),
            static_cast<int>(snapshot->getMax()));
}

}
}

