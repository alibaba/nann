/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * scheduled_reporter.cpp
 *
 *  Created on: Jun 10, 2014
 *      Author: vpoliboy
 */

#include <boost/bind.hpp>
#include "cppmetrics/core/scheduled_reporter.h"

namespace cppmetrics {
namespace core {

ScheduledReporter::ScheduledReporter(MetricRegistryPtr registry,
        boost::chrono::milliseconds rate_unit) :
                running_(false),
                metric_registry_(registry),
                scheduled_executor_(1),
                rate_factor_(
                        boost::chrono::milliseconds(1000).count()
                                / rate_unit.count()),
                duration_factor_(
                        static_cast<double>(1.0)
                                / boost::chrono::duration_cast<
                                        boost::chrono::nanoseconds>(
                                        boost::chrono::milliseconds(1)).count()) {

}

ScheduledReporter::~ScheduledReporter() {
    stop();
}

void ScheduledReporter::report() {
    CounterMap counter_map(metric_registry_->getCounters());
    HistogramMap histogram_map(metric_registry_->getHistograms());
    MeteredMap meter_map(metric_registry_->getMeters());
    TimerMap timer_map(metric_registry_->getTimers());
    GaugeMap gauge_map(metric_registry_->getGauges());
    report(counter_map, histogram_map, meter_map, timer_map, gauge_map);
}

void ScheduledReporter::start(boost::chrono::milliseconds period) {
    if (!running_) {
        running_ = true;
        scheduled_executor_.scheduleAtFixedDelay(
                boost::bind(&ScheduledReporter::report, this), period);
    }
}

void ScheduledReporter::stop() {
    if (running_) {
        running_ = false;
        scheduled_executor_.shutdown();
    }
}

std::string ScheduledReporter::rateUnitInSec() const {
    std::ostringstream ostrstr;
    ostrstr << rate_factor_;
    ostrstr << " Seconds";
    return ostrstr.str();
}

double ScheduledReporter::convertDurationUnit(double duration) const {
    return duration * duration_factor_;
}

double ScheduledReporter::convertRateUnit(double rate) const {
    return rate * rate_factor_;
}

} /* namespace core */
} /* namespace cppmetrics */
