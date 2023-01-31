/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * metered.h
 *
 *  Created on: Jun 4, 2014
 *      Author: vpoliboy
 */

#ifndef METERED_H_
#define METERED_H_

#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include <string>
#include <boost/chrono.hpp>
#include "cppmetrics/core/metric.h"

namespace cppmetrics {
namespace core {

/**
 * Interface for objects which maintains mean and exponentially-weighted rate.
 */
class Metered: public Metric {
public:
    virtual ~Metered() {
    }
    /**
     * @returns the number of events that have been marked.
     */
    virtual boost::uint64_t getCount() const = 0;
    /**
     * @return the fifteen-minute exponentially-weighted moving average rate at which events have
     *         occurred since the meter was created.
     */
    virtual double getFifteenMinuteRate() = 0;
    /**
     * @return the fifteen-minute exponentially-weighted moving average rate at which events have
     *         occurred since the meter was created.
     */
    virtual double getFiveMinuteRate() = 0;
    /**
     * @return the fifteen-minute exponentially-weighted moving average rate at which events have
     *         occurred since the meter was created.
     */
    virtual double getOneMinuteRate() = 0;
    /**
     * @return the average rate at which events have occurred since the meter was created.
     */
    virtual double getMeanRate() = 0;
};

typedef boost::shared_ptr<Metered> MeteredPtr;

} /* namespace core */
} /* namespace cppmetrics */
#endif /* METERED_H_ */
