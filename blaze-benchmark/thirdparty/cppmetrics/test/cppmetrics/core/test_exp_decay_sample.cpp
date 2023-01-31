/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * test_exp_decay_sample.cpp
 *
 *  Created on: Jun 24, 2014
 *      Author: vpoliboy
 */

#include <gtest/gtest.h>
#include <boost/foreach.hpp>
#include "cppmetrics/core/exp_decay_sample.h"

namespace cppmetrics {
namespace core {

namespace {
void assert_all_values_between(SnapshotPtr snapshot, double min, double max) {
	BOOST_FOREACH (double value, snapshot->getValues()) {
		ASSERT_GT(max, value);
		ASSERT_LE(min, value);
	}
}
}

TEST(expdecaysample, sampleSizeOf100OutOf1000ElementsTest) {
	ExpDecaySample exp_decay_sample(100, 0.99);
	for (int i = 0; i < 1000; i++) {
		exp_decay_sample.update(i);
	}

	ASSERT_EQ((size_t)100, exp_decay_sample.size());
	SnapshotPtr snapshot = exp_decay_sample.getSnapshot();

	ASSERT_EQ((size_t)100, snapshot->size());
	assert_all_values_between(snapshot, 0, 1000);
}

TEST(expdecaysample, aHeavilyBiasedSampleOf100OutOf1000Elements) {
	ExpDecaySample exp_decay_sample(1000, 0.01);
	for (int i = 0; i < 100; i++) {
		exp_decay_sample.update(i);
	}

	ASSERT_EQ((size_t)100, exp_decay_sample.size());
	SnapshotPtr snapshot = exp_decay_sample.getSnapshot();

	ASSERT_EQ((size_t)100, snapshot->size());
	assert_all_values_between(snapshot, 0, 100);
}

TEST(expdecaySample, longPeriodsOfInactivityShouldNotCorruptSamplingState) {
	ExpDecaySample exp_decay_sample(10, 0.015);
	Clock::time_point cur_time_point(Clock::now());
    for (int i = 0; i < 1000; i++) {
    	exp_decay_sample.update(1000 + i, cur_time_point);
    	cur_time_point += boost::chrono::milliseconds(100);
    }

	ASSERT_EQ((size_t)10, exp_decay_sample.size());
	SnapshotPtr snapshot = exp_decay_sample.getSnapshot();
	assert_all_values_between(snapshot, 1000, 2000);

	// wait for 15 hours and add another value.
	// this should trigger a rescale. Note that the number of samples will be reduced to 2
	// because of the very small scaling factor that will make all existing priorities equal to
	// zero after rescale.
	cur_time_point += boost::chrono::hours(15);
	exp_decay_sample.update(2000, cur_time_point);
	snapshot = exp_decay_sample.getSnapshot();

	ASSERT_EQ((size_t)2, snapshot->size());
	assert_all_values_between(snapshot, 1000, 3000);

	// add 1000 values at a rate of 10 values/second
	for (int i = 0; i < 1000; i++) {
		exp_decay_sample.update(3000 + i, cur_time_point);
		cur_time_point += boost::chrono::milliseconds(100);
	}
	snapshot = exp_decay_sample.getSnapshot();
	ASSERT_EQ((size_t)10, snapshot->size());
	assert_all_values_between(snapshot, 3000, 4000);
}

}
}

