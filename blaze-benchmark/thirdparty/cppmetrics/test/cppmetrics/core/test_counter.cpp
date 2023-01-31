/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * test_counter.cpp
 *
 *  Created on: Jun 24, 2014
 *      Author: vpoliboy
 */

#include <gtest/gtest.h>
#include "cppmetrics/core/counter.h"

namespace cppmetrics {
namespace core {

TEST(counter, functionaltest) {

    Counter counter;
    ASSERT_EQ(0, counter.getCount());

    counter.increment();
    ASSERT_EQ(1, counter.getCount());

    counter.increment(5);
    ASSERT_EQ(6, counter.getCount());

    counter.increment(3);
    ASSERT_EQ(9, counter.getCount());

    counter.decrement(4);
    ASSERT_EQ(5, counter.getCount());

    counter.decrement();
    ASSERT_EQ(4, counter.getCount());

    counter.setCount(3);
    ASSERT_EQ(3, counter.getCount());

    counter.clear();
    ASSERT_EQ(0, counter.getCount());

    counter.decrement(12);
    ASSERT_EQ(-12, counter.getCount());
}

}
}

