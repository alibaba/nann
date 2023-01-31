/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * timer.cpp
 *
 *  Created on: Jun 5, 2014
 *      Author: vpoliboy
 */

#include "cppmetrics/core/timer.h"

namespace cppmetrics {
namespace core {

Timer::Timer() :
        histogram_(Sampling::kBiased) {
}

Timer::~Timer() {
}

boost::uint64_t Timer::getCount() const {
    return histogram_.getCount();
}

double Timer::getFifteenMinuteRate() {
    return meter_.getFifteenMinuteRate();
}

double Timer::getFiveMinuteRate() {
    return meter_.getFiveMinuteRate();
}

double Timer::getOneMinuteRate() {
    return meter_.getOneMinuteRate();
}

double Timer::getMeanRate() {
    return meter_.getMeanRate();
}

void Timer::clear() {
    histogram_.clear();
}

void Timer::update(boost::chrono::nanoseconds duration) {
    boost::int64_t count = duration.count();
    if (count >= 0) {
        histogram_.update(count);
        meter_.mark();
    }
}

SnapshotPtr Timer::getSnapshot() const {
    return histogram_.getSnapshot();
}

void Timer::time(boost::function<void()> func) {
    TimerContext timer_context(*this);
    func();
}

} /* namespace core */
} /* namespace cppmetrics */
