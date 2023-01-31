/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * test_snapshot.cpp
 *
 *  Created on: Jun 25, 2014
 *      Author: vpoliboy
 */

#include <gtest/gtest.h>
#include "cppmetrics/core/snapshot.h"

namespace cppmetrics {
namespace core {

TEST(snapshot, zeroSampleTest) {
    const ValueVector samples;
    Snapshot snapshot(samples);
    ASSERT_EQ(0, snapshot.getMin());
    ASSERT_EQ(0, snapshot.getMax());
    ASSERT_NEAR(0.0, snapshot.getMean(), 0.000001);
    ASSERT_NEAR(0.0, snapshot.getStdDev(), 0.0001);
}

TEST(snapshot, oneSampleTest) {
    const ValueVector::value_type values_array[] = { 1 };
    const size_t element_count(sizeof(values_array) / sizeof(values_array[0]));
    const ValueVector values(values_array, values_array + element_count);
    Snapshot snapshot(values);
    ASSERT_EQ(1, snapshot.getMin());
    ASSERT_EQ(1, snapshot.getMax());
    ASSERT_NEAR(1.0, snapshot.getMean(), 0.000001);
    ASSERT_NEAR(0.0, snapshot.getStdDev(), 0.0001);
}

TEST(snapshot, minMaxMedianPercentileWithFiveSamplesTest) {

    const ValueVector::value_type values_array[] = { 5, 1, 2, 3, 4 };
    const size_t element_count(sizeof(values_array) / sizeof(values_array[0]));
    const ValueVector values(values_array, values_array + element_count);
    Snapshot snapshot(values);

    ASSERT_EQ(element_count, snapshot.size());

    ValueVector expected_values(values);
    std::sort(expected_values.begin(), expected_values.end());
    ASSERT_EQ(expected_values, snapshot.getValues());

    ASSERT_NEAR(1.0, snapshot.getValue(0.0), 0.1);
    ASSERT_NEAR(5.0, snapshot.getValue(1.0), 0.1);
    ASSERT_NEAR(3.0, snapshot.getMedian(), 0.1);
    ASSERT_NEAR(4.5, snapshot.get75thPercentile(), 0.1);
    ASSERT_NEAR(5.0, snapshot.get95thPercentile(), 0.1);
    ASSERT_NEAR(5.0, snapshot.get98thPercentile(), 0.1);
    ASSERT_NEAR(5.0, snapshot.get99thPercentile(), 0.1);
    ASSERT_NEAR(5.0, snapshot.get99thPercentile(), 0.1);
    ASSERT_EQ(1, snapshot.getMin());
    ASSERT_EQ(5, snapshot.getMax());
    ASSERT_NEAR(3.0, snapshot.getMean(), 0.000001);
    ASSERT_NEAR(1.5811, snapshot.getStdDev(), 0.0001);

}

}
}

