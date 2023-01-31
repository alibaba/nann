/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * console_reporter.h
 *
 *  Created on: Jul 1, 2014
 *      Author: vpoliboy
 */

#ifndef CONSOLE_REPORTER_H_
#define CONSOLE_REPORTER_H_

#include "cppmetrics/core/scheduled_reporter.h"

namespace cppmetrics {
namespace core {

/*
 *  A GLOG console reporter that periodically logs the metric values.
 */
class ConsoleReporter: public ScheduledReporter, boost::noncopyable {
public:

    /**
     * Creates a {@link ConsoleReporter} instance. Uses the given registry.
     * @param registry The metric registry.
     * @param ostr The output stream used for printing the values.
     * @param rate_unit The conversion unit user for the rate metrics.
     */
    ConsoleReporter(MetricRegistryPtr registry,
            std::ostream& ostr,
            boost::chrono::milliseconds rate_unit = boost::chrono::seconds(1));
    virtual ~ConsoleReporter();

    /**
     * Reports all the metrics from the registry periodically to the graphite server.
     * @param gauge_map     all of the gauges in the registry
     * @param counter_map   all of the counters in the registry
     * @param histogram_map all of the histograms in the registry
     * @param meter_map     all of the meters in the registry
     * @param timer_map     all of the timers in the registry
     */
    virtual void report(core::CounterMap counter_map,
            core::HistogramMap histogram_map,
            core::MeteredMap meter_map,
            core::TimerMap timer_map,
            core::GaugeMap gauge_map);

private:
    void printWithBanner(const std::string& s, char sep);

    void printGauge(const core::GaugeMap::mapped_type& metric);

    void printCounter(const core::CounterMap::mapped_type& metric);

    void printHistogram(const core::HistogramMap::mapped_type& metric);

    void printTimer(const core::TimerMap::mapped_type& metric);

    void printMeter(const core::MeteredMap::mapped_type& meter);

    static const size_t CONSOLE_WIDTH = 80;

    std::ostream& ostr_;

};

} /* namespace core */
} /* namespace cppmetrics */
#endif /* CONSOLER_REPORTER_H_ */
