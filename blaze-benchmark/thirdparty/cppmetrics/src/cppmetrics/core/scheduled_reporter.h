/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * scheduled_reporter.h
 *
 *  Created on: Jun 10, 2014
 *      Author: vpoliboy
 */

#ifndef SCHEDULED_REPORTER_H_
#define SCHEDULED_REPORTER_H_

#include "reporter.h"
#include "cppmetrics/concurrent/simple_scheduled_thread_pool_executor.h"
#include "cppmetrics/core/metric_registry.h"

namespace cppmetrics {
namespace core {

/**
 * The abstract base class for all scheduled reporters (i.e., reporters which process a registry's
 * metrics periodically).
 */
class ScheduledReporter: public Reporter {
public:
    virtual ~ScheduledReporter();

    /**
     * Report the current values of all metrics in the registry.
     */
    virtual void report();

    /**
     * Called periodically by the polling thread. Subclasses should report all the given metrics.
     * @param gauge_map     all of the gauges in the registry
     * @param counter_map   all of the counters in the registry
     * @param histogram_map all of the histograms in the registry
     * @param meter_map     all of the meters in the registry
     * @param timer_map     all of the timers in the registry
     */
    virtual void report(CounterMap counter_map,
            HistogramMap histogram_map,
            MeteredMap meter_map,
            TimerMap timer_map,
            GaugeMap gauge_map) = 0;

    /**
     * Starts a background thread which polls and published the metrics from the registry periodically at the given
     *  interval.
     * @param period the amount of time between polls in milliseconds.
     */
    virtual void start(boost::chrono::milliseconds period);

    /**
     * Shuts down the background thread that polls/publishes the metrics from the registry.
     */
    virtual void stop();

protected:

    /**
     * Creates a new {@link ScheduledReporter} instance.
     * @param registry the {@link MetricRegistry} shared_ptr containing the metrics this
     *                 reporter will report
     * @param rate_unit a unit of time used for publishing the rate metrics like meter.
     */
    ScheduledReporter(MetricRegistryPtr registry,
            boost::chrono::milliseconds rate_unit);

    /**
     * Converts the duration value to the milliseconds (from nanoseconds).
     * @param duration_value The duration value from a metric like timer.
     * @return The converted duration value in terms number of millis.
     */
    double convertDurationUnit(double duration_value) const;

    /**
     * Converts the rate value based on the unit of duration.
     * @param rate_value The duration value from a metric like meter.
     * @return The converted rate value.
     */
    double convertRateUnit(double rate_value) const;

    /**
     *
     * @returns the Rate unit in seconds in string format.
     */
    std::string rateUnitInSec() const;

private:
    bool running_;
    MetricRegistryPtr metric_registry_;
    concurrent::SimpleScheduledThreadPoolExecutor scheduled_executor_;
    double rate_factor_;
    double duration_factor_;
};

} /* namespace core */
} /* namespace cppmetrics */
#endif /* SCHEDULED_REPORTER_H_ */
