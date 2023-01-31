/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * console_reporter.cpp
 *
 *  Created on: Jul 1, 2014
 *      Author: vpoliboy
 */

#include <glog/logging.h>
#include "console_reporter.h"
#include "utils.h"

namespace cppmetrics {
namespace core {

ConsoleReporter::ConsoleReporter(MetricRegistryPtr registry,
        std::ostream& ostr,
        boost::chrono::milliseconds rate_unit) :
        ScheduledReporter(registry, rate_unit), ostr_(ostr) {
    ostr_.setf(std::ios_base::fixed, std::ios_base::floatfield);
    ostr_.width(2);
    ostr_.precision(2);
}

ConsoleReporter::~ConsoleReporter() {
}

void ConsoleReporter::report(core::CounterMap counter_map,
        core::HistogramMap histogram_map,
        core::MeteredMap meter_map,
        core::TimerMap timer_map,
        core::GaugeMap gauge_map) {

    std::string timestamp = utc_timestamp(ostr_.getloc());
    printWithBanner(timestamp, '=');

    if (!gauge_map.empty()) {
        printWithBanner("-- Gauges", '-');
        BOOST_FOREACH(const core::GaugeMap::value_type& entry, gauge_map) {
            ostr_ << entry.first << std::endl;
            printGauge(entry.second);
        }
        ostr_ << std::endl;
    }

    if (!counter_map.empty()) {
        printWithBanner("-- Counters", '-');
        BOOST_FOREACH(const core::CounterMap::value_type& entry, counter_map){
            ostr_ << entry.first << std::endl;
            printCounter(entry.second);
        }
        ostr_ << std::endl;
    }

    if (!histogram_map.empty()) {
        printWithBanner("-- Histograms", '-');
        BOOST_FOREACH(const core::HistogramMap::value_type& entry, histogram_map) {
            ostr_ << entry.first << std::endl;
            printHistogram(entry.second);
        }
        ostr_ << std::endl;
    }

    if (!meter_map.empty()) {
        printWithBanner("-- Meters", '-');
        BOOST_FOREACH(const core::MeteredMap::value_type& entry, meter_map) {
            ostr_ << entry.first << std::endl;
            printMeter(entry.second);
        }
        ostr_ << std::endl;
    }

    if (!timer_map.empty()) {
        printWithBanner("-- Timers", '-');
        BOOST_FOREACH(const core::TimerMap::value_type& entry, timer_map) {
            ostr_ << entry.first << std::endl;
            printTimer(entry.second);
        }
        ostr_ << std::endl;
    }
    ostr_ << std::endl;
    ostr_.flush();
}

void ConsoleReporter::printMeter(const core::MeteredMap::mapped_type& meter) {
    ostr_ << "             count = " << meter->getCount() << std::endl;
    ostr_ << "         mean rate = " << convertRateUnit(meter->getMeanRate())
            << " events per " << rateUnitInSec() << std::endl;
    ostr_ << "     1-minute rate = "
            << convertRateUnit(meter->getOneMinuteRate()) << " events per "
            << rateUnitInSec() << std::endl;
    ostr_ << "     5-minute rate = "
            << convertRateUnit(meter->getFiveMinuteRate()) << " events per "
            << rateUnitInSec() << std::endl;
    ostr_ << "    15-minute rate = "
            << convertRateUnit(meter->getFifteenMinuteRate()) << " events per "
            << rateUnitInSec() << std::endl;
}

void ConsoleReporter::printCounter(const core::CounterMap::mapped_type& counter_ptr) {
    ostr_ << "             count = " << counter_ptr->getCount() << std::endl;
}

void ConsoleReporter::printGauge(const core::GaugeMap::mapped_type& gauge_ptr) {
    ostr_ << "             value = " << gauge_ptr->getValue() << std::endl;
}

void ConsoleReporter::printHistogram(const core::HistogramMap::mapped_type& histogram_ptr) {
    ostr_ << "             count = " << histogram_ptr->getCount() << std::endl;
    SnapshotPtr snapshot = histogram_ptr->getSnapshot();
    ostr_ << "               min = " << snapshot->getMin() << std::endl;
    ostr_ << "               max = " << snapshot->getMax() << std::endl;
    ostr_ << "              mean = " << snapshot->getMean() << std::endl;
    ostr_ << "            stddev = " << snapshot->getStdDev() << std::endl;
    ostr_ << "            median = " << snapshot->getMedian() << std::endl;
    ostr_ << "              75% <= " << snapshot->get75thPercentile() << std::endl;
    ostr_ << "              95% <= " << snapshot->get95thPercentile() << std::endl;
    ostr_ << "              98% <= " << snapshot->get98thPercentile() << std::endl;
    ostr_ << "              99% <= " << snapshot->get99thPercentile() << std::endl;
    ostr_ << "            99.9% <= " << snapshot->get999thPercentile() << std::endl;
}

void ConsoleReporter::printTimer(const core::TimerMap::mapped_type& timer) {
    SnapshotPtr snapshot = timer->getSnapshot();
    ostr_ << "             count = " << timer->getCount() << std::endl;
    ostr_ << "         mean rate = " << convertRateUnit(timer->getMeanRate())
    		<< " calls per " << rateUnitInSec() << std::endl;
    ostr_ << "     1-minute rate = " << convertRateUnit(timer->getOneMinuteRate())
    		<< " calls per " << rateUnitInSec() << std::endl;
    ostr_ << "     5-minute rate = " << convertRateUnit(timer->getFiveMinuteRate())
    		<< " calls per " << rateUnitInSec() << std::endl;
    ostr_ << "    15-minute rate = " << convertRateUnit(timer->getFifteenMinuteRate())
    		<< " calls per " << rateUnitInSec() << std::endl;
    ostr_ << "               min = " << convertDurationUnit(snapshot->getMin()) << " millis " << std::endl;
    ostr_ << "               max = " << convertDurationUnit(snapshot->getMax()) << " millis " << std::endl;
    ostr_ << "              mean = " << convertDurationUnit(snapshot->getMean()) << " millis " << std::endl;
    ostr_ << "            stddev = " << convertDurationUnit(snapshot->getStdDev()) << " millis " << std::endl;
    ostr_ << "            median = " << convertDurationUnit(snapshot->getMedian()) << " millis "<< std::endl;
    ostr_ << "              75% <= " << convertDurationUnit(snapshot->get75thPercentile()) << " millis " << std::endl;
    ostr_ << "              95% <= " << convertDurationUnit(snapshot->get95thPercentile()) << " millis " << std::endl;
    ostr_ << "              98% <= " << convertDurationUnit(snapshot->get98thPercentile()) << " millis " << std::endl;
    ostr_ << "              99% <= " << convertDurationUnit(snapshot->get99thPercentile()) << " millis " << std::endl;
    ostr_ << "            99.9% <= " << convertDurationUnit(snapshot->get999thPercentile())<< " millis " << std::endl;
}

void ConsoleReporter::printWithBanner(const std::string& s, char sep) {
    ostr_ << s << ' ';
    for (size_t i = 0; i < (CONSOLE_WIDTH - s.size() - 1); i++) {
        ostr_ << sep;
    }
    ostr_ << std::endl;
}

} /* namespace core */
} /* namespace cppmetrics */
