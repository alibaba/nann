/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * test_simple_thread_pool_executor.cpp
 *
 *  Created on: Jun 26, 2014
 *      Author: vpoliboy
 */

#include <gtest/gtest.h>
#include "cppmetrics/concurrent/simple_thread_pool_executor.h"

namespace cppmetrics {
namespace concurrent {

namespace {
void increment_counter(size_t& counter) {
    ++counter;
}
}

TEST(simplethreadpoolexecutor, functionaltest) {
    SimpleThreadPoolExecutor thread_pool_executor(2);

    ASSERT_FALSE(thread_pool_executor.isShutdown());
    size_t counter = 0;
    boost::function<void()> task(
            boost::bind(increment_counter, boost::ref(counter)));
    thread_pool_executor.execute(task);
    // Give up a timeslice.
    boost::this_thread::sleep(boost::posix_time::milliseconds(20));
    ASSERT_EQ((size_t )1, counter);
    ASSERT_FALSE(thread_pool_executor.isShutdown());
    thread_pool_executor.shutdownNow();
    ASSERT_TRUE(thread_pool_executor.isShutdown());
}

}
}

