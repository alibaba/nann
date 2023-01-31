/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * test_console_reporter.cpp
 *
 *  Created on: Jul 2, 2014
 *      Author: vpoliboy
 */

#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/thread.hpp>
#include <boost/foreach.hpp>
#include "cppmetrics/core/console_reporter.h"

namespace cppmetrics {
namespace core {

namespace {

class TestGauge: public Gauge {
public:
    virtual boost::int64_t getValue() {
        return 100;
    }
};

}

TEST(consolereporter, gaugetest) {

    MetricRegistryPtr metric_registry(new MetricRegistry());
    metric_registry->addGauge("new.gauge", GaugePtr(new TestGauge()));
    ConsoleReporter console_reporter(metric_registry, std::cout);
    console_reporter.start(boost::chrono::milliseconds(1000));
    boost::this_thread::sleep(boost::posix_time::milliseconds(10 * 1000));
}

TEST(consolereporter, timerContextTest) {

    MetricRegistryPtr metric_registry(new MetricRegistry());
    boost::mt11213b rng;
    ConsoleReporter console_reporter(metric_registry, std::cout);
    console_reporter.start(boost::chrono::milliseconds(1000));
    for (size_t i = 0; i < 100; ++i) {
        boost::random::uniform_int_distribution<> uniform(10, 30);
        size_t sleep_time = uniform(rng);
        TimerContextPtr time_context(
                metric_registry->timer("test.timer")->timerContextPtr());
        boost::this_thread::sleep(boost::posix_time::milliseconds(sleep_time));
    }
    boost::this_thread::sleep(boost::posix_time::milliseconds(10 * 1000));
}

}
}

