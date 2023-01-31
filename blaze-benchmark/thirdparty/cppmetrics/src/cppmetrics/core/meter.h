/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * meter.h
 *
 *  Created on: Jun 4, 2014
 *      Author: vpoliboy
 */

#ifndef METER_H_
#define METER_H_

#include <boost/chrono.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/atomic.hpp>
#include "cppmetrics/core/metered.h"

namespace cppmetrics {
namespace core {

/**
 * A meter metric which measures mean throughput and one-, five-, and fifteen-minute
 * exponentially-weighted moving average throughputs.
 */
class Meter: public Metered {
public:
    /**
     * Creates a meter with the specified rate unit.
     * @param rate_unit The rate unit in nano seconds.
     */
    Meter(boost::chrono::nanoseconds rate_unit = boost::chrono::seconds(1));

    virtual ~Meter();

    /**
     * @returns the number of events that have been marked.
     */
    virtual boost::uint64_t getCount() const;

    /**
     * @return the fifteen-minute exponentially-weighted moving average rate at which events have
     *         occurred since the meter was created.
     */
    virtual double getFifteenMinuteRate();

    /**
     * @return the five-minute exponentially-weighted moving average rate at which events have
     *         occurred since the meter was created.
     */
    virtual double getFiveMinuteRate();

    /**
     * @return the one-minute exponentially-weighted moving average rate at which events have
     *         occurred since the meter was created.
     */
    virtual double getOneMinuteRate();

    /**
     * @return the mean rate at which events have occurred since the meter was created.
     */
    virtual double getMeanRate();

    /**
     * Mark the occurrence of a given number of events.
     * @param n the number of events with the default being 1.
     */
    void mark(boost::uint64_t n = 1);

private:
    class Impl;
    boost::scoped_ptr<Impl> impl_;
};

typedef boost::shared_ptr<Meter> MeterPtr;

} /* namespace core */
} /* namespace cppmetrics */
#endif /* METER_H_ */
