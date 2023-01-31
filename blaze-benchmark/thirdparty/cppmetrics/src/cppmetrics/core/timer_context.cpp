/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * timer_context.cpp
 *
 *  Created on: Jun 5, 2014
 *      Author: vpoliboy
 */

#include "cppmetrics/core/timer_context.h"
#include "cppmetrics/core/timer.h"

namespace cppmetrics {
namespace core {

TimerContext::TimerContext(Timer& timer) :
        timer_(timer) {
    reset();
}

TimerContext::~TimerContext() {
    stop();
}

void TimerContext::reset() {
    active_ = true;
    start_time_ = Clock::now();
}

boost::chrono::nanoseconds TimerContext::stop() {
    if (active_) {
        boost::chrono::nanoseconds dur = Clock::now() - start_time_;
        timer_.update(dur);
        active_ = false;
        return dur;
    }
    return boost::chrono::nanoseconds(0);
}

} /* namespace core */
} /* namespace cppmetrics */
