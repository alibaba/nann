/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * test_histogram.cpp
 *
 *  Created on: Jun 24, 2014
 *      Author: vpoliboy
 */

#include <gtest/gtest.h>
#include "cppmetrics/core/histogram.h"

namespace cppmetrics {
namespace core {

TEST(histogram, updatesTheCountOnUpdatesTest) {
    Histogram histogram;
    ASSERT_EQ((size_t )0, histogram.getCount());
    ASSERT_EQ((size_t )0, histogram.getSnapshot()->size());

    histogram.update(1);
    ASSERT_EQ((size_t )1, histogram.getCount());
    ASSERT_EQ((size_t )1, histogram.getSnapshot()->size());

    const size_t update_count = 4096;
    for (size_t i = 0; i < update_count; ++i) {
        histogram.update(i);
    }

    const size_t max_sample_size = 1024;
    ASSERT_EQ(update_count + 1, histogram.getCount());
    ASSERT_EQ(max_sample_size, histogram.getSnapshot()->size());
}

}
}

