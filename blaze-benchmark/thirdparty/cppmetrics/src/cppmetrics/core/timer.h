/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * timer.h
 *
 *  Created on: Jun 5, 2014
 *      Author: vpoliboy
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <string>
#include <boost/chrono.hpp>
#include <boost/cstdint.hpp>
#include <boost/function.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include "cppmetrics/core/metered.h"
#include "cppmetrics/core/metric.h"
#include "cppmetrics/core/sampling.h"
#include "cppmetrics/core/timer_context.h"
#include "cppmetrics/core/meter.h"
#include "cppmetrics/core/histogram.h"

namespace cppmetrics {
namespace core {

/**
 * A timer metric which aggregates timing durations and provides duration statistics, plus
 * throughput statistics via {@link Meter} and {@link Histogram}.
 */
class Timer: public Metered, Sampling {
public:
    /**
     * Creates a new {@link Timer} using an {@link ExpDecaySample}.
     */
    Timer();
    virtual ~Timer();

    /**
     * @returns the number of events that have been measured.
     */
    virtual boost::uint64_t getCount() const;

    /**
     * @return the fifteen-minute exponentially-weighted moving average rate at which events have
     *         occurred since the timer was created.
     */
    virtual double getFifteenMinuteRate();

    /**
     * @return the five-minute exponentially-weighted moving average rate at which events have
     *         occurred since the timer was created.
     */
    virtual double getFiveMinuteRate();

    /**
     * @return the one-minute exponentially-weighted moving average rate at which events have
     *         occurred since the timer was created.
     */
    virtual double getOneMinuteRate();

    /**
     * @return the average rate at which events have occurred since the meter was created.
     */
    virtual double getMeanRate();

    /**
     * @return the current snapshot based on the sample.
     */
    virtual SnapshotPtr getSnapshot() const;

    /**
     * Clears the underlying metrics.
     */
    void clear();

    /**
     * Adds a recorded duration.
     * @param duration the length of the duration in nanos.
     */
    void update(boost::chrono::nanoseconds duration);

    /**
     * Creates a new TimerContext instance that measures the duration and updates the
     * duration before the instance goes out of scope.
     * @return The TimerContext object.
     * @note The TimerContextPtr should not be shared.
     */
    TimerContextPtr timerContextPtr() {
        return boost::shared_ptr<TimerContext>(new TimerContext(*this));
    }

    /**
     * Times the duration of a function that will be executed internally and updates the duration.
     * @param The fn to be timed.
     */
    void time(boost::function<void()> fn);

private:
    Meter meter_; /**< The underlying meter object */
    Histogram histogram_; /**< The underlying histogram object */
};

typedef boost::shared_ptr<Timer> TimerPtr;

} /* namespace core */
} /* namespace cppmetrics */
#endif /* TIMER_H_ */
