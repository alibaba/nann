/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * test_metric_registry.cpp
 *
 *  Created on: Jun 25, 2014
 *      Author: vpoliboy
 */

#include <gtest/gtest.h>
#include "cppmetrics/core/metric_registry.h"

namespace cppmetrics {
namespace core {

namespace {

class FakeGauge: public Gauge {
public:
    virtual ~FakeGauge() {
    }
    ;
    virtual boost::int64_t getValue() {
        return 10;
    }
};

}

TEST(metricregistry, initialTest) {
    MetricRegistry metric_registry;
    ASSERT_EQ((size_t )0, metric_registry.count());
    ASSERT_EQ((size_t )0, metric_registry.getCounters().size());
    ASSERT_EQ((size_t )0, metric_registry.getGauges().size());
    ASSERT_EQ((size_t )0, metric_registry.getHistograms().size());
    ASSERT_EQ((size_t )0, metric_registry.getMeters().size());
    ASSERT_EQ((size_t )0, metric_registry.getTimers().size());

    ASSERT_FALSE(metric_registry.removeMetric("random_metric_name"));
}

TEST(metricregistry, counterTest) {
    MetricRegistry metric_registry;

    const std::string counter1("counter1");
    const std::string counter2("counter2");

    // Create a counter
    CounterPtr counter_ptr1(metric_registry.counter(counter1));
    ASSERT_TRUE(counter_ptr1.get() != NULL);

    // Fetch already created counter.
    CounterPtr counter_ptr2(metric_registry.counter(counter1));
    ASSERT_TRUE(counter_ptr1.get() == counter_ptr2.get());

    // Cannot add a new metric type with existing type.
    ASSERT_THROW(metric_registry.histogram(counter1), std::invalid_argument);

    // Create another counter
    CounterPtr counter_ptr3(metric_registry.counter(counter2));
    ASSERT_TRUE(counter_ptr3.get() != NULL);
    ASSERT_TRUE(counter_ptr1.get() != counter_ptr3.get());

    CounterMap counters(metric_registry.getCounters());
    ASSERT_EQ((size_t )2, counters.size());
    ASSERT_STREQ(counter1.c_str(), counters.begin()->first.c_str());
    ASSERT_STREQ(counter2.c_str(), counters.rbegin()->first.c_str());

    ASSERT_TRUE(metric_registry.removeMetric(counter2));
    counters = metric_registry.getCounters();
    ASSERT_EQ((size_t )1, counters.size());
    ASSERT_STREQ(counter1.c_str(), counters.begin()->first.c_str());
}

TEST(metricregistry, gaugeTest) {
    MetricRegistry metric_registry;

    const std::string gauge1("gauge1");
    const std::string gauge2("gauge2");

    // Create a gauge
    GaugePtr gauge_ptr1(new FakeGauge());
    ASSERT_TRUE(metric_registry.addGauge(gauge1, gauge_ptr1));

    // Create another gauge
    GaugePtr gauge_ptr2(new FakeGauge());
    // Cannot add a new gauge with same name.
    ASSERT_THROW(metric_registry.addGauge(gauge1, gauge_ptr2),
            std::invalid_argument);
    // Try creating a different metric with the same name.
    ASSERT_THROW(metric_registry.counter(gauge1), std::invalid_argument);

    // add a new gauge with different name.
    ASSERT_TRUE(metric_registry.addGauge(gauge2, gauge_ptr2));
    GaugeMap gauges(metric_registry.getGauges());
    ASSERT_EQ((size_t )2, gauges.size());
    ASSERT_STREQ(gauge1.c_str(), gauges.begin()->first.c_str());
    ASSERT_STREQ(gauge2.c_str(), gauges.rbegin()->first.c_str());

    ASSERT_TRUE(metric_registry.removeMetric(gauge1));
    ASSERT_FALSE(metric_registry.removeMetric(gauge1));
    gauges = metric_registry.getGauges();
    ASSERT_EQ((size_t )1, gauges.size());
    ASSERT_STREQ(gauge2.c_str(), gauges.begin()->first.c_str());
}

TEST(metricregistry, histogramTest) {
    MetricRegistry metric_registry;

    const std::string histogram1("histogram1");
    const std::string histogram2("histogram2");

    // Create a histogram
    HistogramPtr histogram_ptr1(metric_registry.histogram(histogram1));
    ASSERT_TRUE(histogram_ptr1.get() != NULL);

    // Fetch already created histogram.
    HistogramPtr histogram_ptr2(metric_registry.histogram(histogram1));
    ASSERT_TRUE(histogram_ptr1.get() == histogram_ptr2.get());

    // Cannot add a new metric type with existing type.
    ASSERT_THROW(metric_registry.counter(histogram1), std::invalid_argument);

    // Create another histogram
    HistogramPtr histogram_ptr3(metric_registry.histogram(histogram2));
    ASSERT_TRUE(histogram_ptr3.get() != NULL);
    ASSERT_TRUE(histogram_ptr1.get() != histogram_ptr3.get());

    HistogramMap histograms(metric_registry.getHistograms());
    ASSERT_EQ((size_t )2, histograms.size());
    ASSERT_STREQ(histogram1.c_str(), histograms.begin()->first.c_str());
    ASSERT_STREQ(histogram2.c_str(), histograms.rbegin()->first.c_str());

    ASSERT_TRUE(metric_registry.removeMetric(histogram2));
    histograms = metric_registry.getHistograms();
    ASSERT_EQ((size_t )1, histograms.size());
    ASSERT_STREQ(histogram1.c_str(), histograms.begin()->first.c_str());
}

TEST(metricregistry, meterTest) {
    MetricRegistry metric_registry;

    const std::string meter1("meter1");
    const std::string meter2("meter2");

    // Create a meter
    MeterPtr meter_ptr1(metric_registry.meter(meter1));
    ASSERT_TRUE(meter_ptr1.get() != NULL);

    // Fetch already created meter.
    MeterPtr meter_ptr2(metric_registry.meter(meter1));
    ASSERT_TRUE(meter_ptr1.get() == meter_ptr2.get());

    // Cannot add a new metric type with existing type.
    ASSERT_THROW(metric_registry.counter(meter1), std::invalid_argument);

    // Create another meter
    MeterPtr meter_ptr3(metric_registry.meter(meter2));
    ASSERT_TRUE(meter_ptr3.get() != NULL);
    ASSERT_TRUE(meter_ptr1.get() != meter_ptr3.get());

    MeteredMap meters(metric_registry.getMeters());
    ASSERT_EQ((size_t )2, meters.size());
    ASSERT_STREQ(meter1.c_str(), meters.begin()->first.c_str());
    ASSERT_STREQ(meter2.c_str(), meters.rbegin()->first.c_str());

    ASSERT_TRUE(metric_registry.removeMetric(meter2));
    meters = metric_registry.getMeters();
    ASSERT_EQ((size_t )1, meters.size());
    ASSERT_STREQ(meter1.c_str(), meters.begin()->first.c_str());
}

TEST(metricregistry, timerTest) {
    MetricRegistry metric_registry;

    const std::string timer1("timer1");
    const std::string timer2("timer2");

    // Create a timer
    TimerPtr timer_ptr1(metric_registry.timer(timer1));
    ASSERT_TRUE(timer_ptr1.get() != NULL);

    // Fetch already created timer.
    TimerPtr timer_ptr2(metric_registry.timer(timer1));
    ASSERT_TRUE(timer_ptr1.get() == timer_ptr2.get());

    // Cannot add a new metric type with existing type.
    ASSERT_THROW(metric_registry.counter(timer1), std::invalid_argument);

    // Create another timer
    TimerPtr timer_ptr3(metric_registry.timer(timer2));
    ASSERT_TRUE(timer_ptr3.get() != NULL);
    ASSERT_TRUE(timer_ptr1.get() != timer_ptr3.get());

    TimerMap timers(metric_registry.getTimers());
    ASSERT_EQ((size_t )2, timers.size());
    ASSERT_STREQ(timer1.c_str(), timers.begin()->first.c_str());
    ASSERT_STREQ(timer2.c_str(), timers.rbegin()->first.c_str());

    ASSERT_TRUE(metric_registry.removeMetric(timer2));
    timers = metric_registry.getTimers();
    ASSERT_EQ((size_t )1, timers.size());
    ASSERT_STREQ(timer1.c_str(), timers.begin()->first.c_str());
}

}
}

