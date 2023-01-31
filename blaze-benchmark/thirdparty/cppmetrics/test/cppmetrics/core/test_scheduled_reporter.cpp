/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * test_scheduled_reporter.cpp
 *
 *  Created on: Jun 25, 2014
 *      Author: vpoliboy
 */

#include <gtest/gtest.h>
#include <iostream>
#include "cppmetrics/core/scheduled_reporter.h"

namespace cppmetrics {
namespace core {

namespace {

class StubScheduledReporter: public ScheduledReporter {
public:
	StubScheduledReporter(MetricRegistryPtr registry, boost::chrono::milliseconds rate_unit) :
			ScheduledReporter(registry, rate_unit),
			invocation_count_(0) {
		last_time_ = Clock::now();
	}

	virtual ~StubScheduledReporter() {
	}

	size_t invocation_count() const {
		return invocation_count_;
	}

	virtual void report(CounterMap counter_map,
			HistogramMap histogram_map,
			MeteredMap meter_map,
			TimerMap timer_map,
			GaugeMap gauge_map) {
		++invocation_count_;
		boost::chrono::milliseconds invocation_period(
				boost::chrono::duration_cast<boost::chrono::milliseconds>(Clock::now() - last_time_));
		std::cout << invocation_count_ << " Invocation period(in millis): " << invocation_period.count()
				  << std::endl;
		last_time_ = Clock::now();
	}

private:
	size_t invocation_count_;
	Clock::time_point last_time_;
};

}

TEST(scheduledreporter, test) {
	StubScheduledReporter scheduled_reporter(MetricRegistry::DEFAULT_REGISTRY(),
			boost::chrono::milliseconds(1));
	scheduled_reporter.start(boost::chrono::milliseconds(100));
	boost::this_thread::sleep(boost::posix_time::seconds(1));
	scheduled_reporter.stop();
	ASSERT_LE((size_t)9, scheduled_reporter.invocation_count());
	ASSERT_GE((size_t)11, scheduled_reporter.invocation_count());
}

}
}


