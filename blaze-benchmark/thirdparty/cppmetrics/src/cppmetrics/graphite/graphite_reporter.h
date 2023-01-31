/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * graphite_reporter.h
 *
 *  Created on: Jun 12, 2014
 *      Author: vpoliboy
 */

#ifndef GRAPHITE_REPORTER_H_
#define GRAPHITE_REPORTER_H_

#include <boost/scoped_ptr.hpp>
#include <boost/noncopyable.hpp>
#include <boost/chrono.hpp>
#include "cppmetrics/core/scheduled_reporter.h"
#include "cppmetrics/graphite/graphite_sender.h"

namespace cppmetrics {
namespace graphite {

/**
 * A reporter which publishes metric values to a Graphite server.
 * @see <a href="http://graphite.wikidot.com/">Graphite - Scalable Realtime Graphing</a>
 */
class GraphiteReporter: public core::ScheduledReporter, boost::noncopyable {
public:
    /**
     * Creates a {@link GraphiteReporter} instance. Uses the given registry, metricname prefix.
     * @param registry The metric registry.
     * @param graphite_sender The graphite server sender.
     * @param prefix The prefix thats added to all the metric names in the registry before posting to graphite.
     * @param rateUnit The conversion unit user for the rate metrics.
     * @param durationUnit The conversion unit used for the duration metrics.
     */
    GraphiteReporter(core::MetricRegistryPtr registry,
            GraphiteSenderPtr graphite_sender,
            std::string prefix,
            boost::chrono::milliseconds rateUnit = boost::chrono::seconds(1));
    virtual ~GraphiteReporter();

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

    std::string prefix(const std::string& name, const char* extra = NULL);

    template<class T> std::string format(T o);

    void reportTimer(const std::string& name,
            core::TimerPtr timer,
            boost::uint64_t timestamp);

    void reportMeter(const std::string& name,
            core::MeteredPtr meter,
            boost::uint64_t timestamp);

    void reportHistogram(const std::string& name,
            core::HistogramPtr histogram,
            boost::uint64_t timestamp);

    void reportCounter(const std::string& name,
            core::CounterPtr counter,
            boost::uint64_t timestamp);

    void reportGauge(const std::string& name,
            core::GaugePtr gauge,
            boost::uint64_t timestamp);

    core::MetricRegistryPtr registry_;
    GraphiteSenderPtr sender_;
    std::string prefix_;
    boost::chrono::milliseconds rate_unit_;
    boost::chrono::milliseconds duration_unit_;
};

} /* namespace graphite */
} /* namespace cppmetrics */
#endif /* GRAPHITE_REPORTER_H_ */
