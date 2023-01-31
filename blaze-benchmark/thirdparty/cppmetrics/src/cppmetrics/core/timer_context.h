/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * timer_context.h
 *
 *  Created on: Jun 5, 2014
 *      Author: vpoliboy
 */

#ifndef TIMER_CONTEXT_H_
#define TIMER_CONTEXT_H_

#include <boost/chrono.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include "cppmetrics/core/types.h"

namespace cppmetrics {
namespace core {

class Timer;

/**
 * Class that actually measures the wallclock time.
 */
class TimerContext {
public:

    /**
     * Creates a TimerContext.
     * @param timer The parent timer metric.
     */
    TimerContext(Timer& timer);

    ~TimerContext();

    /**
     * Resets the underlying clock.
     */
    void reset();

    /**
     * Stops recording the elapsed time and updates the timer.
     * @return the elapsed time in nanoseconds
     */
    boost::chrono::nanoseconds stop();
private:

    TimerContext& operator=(const TimerContext&);

    Clock::time_point start_time_; ///< The start time on instantitation */
    Timer& timer_;                 ///< The parent timer object. */
    bool active_;                  ///< Whether the timer is active or not */
};

typedef boost::shared_ptr<TimerContext> TimerContextPtr;

} /* namespace cppmetrics */
} /* namespace core */
#endif /* TIMER_CONTEXT_H_ */
