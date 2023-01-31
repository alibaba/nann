/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * counter.h
 *
 *  Created on: Jun 5, 2014
 *      Author: vpoliboy
 */

#ifndef COUNTER_H_
#define COUNTER_H_

#include <boost/cstdint.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/atomic.hpp>
#include "cppmetrics/core/metric.h"

namespace cppmetrics {
namespace core {

/**
 * An incrementing and decrementing counter metric.
 * thread-safety: thread safe
 */
class Counter: public Metric {
public:

    /**
     * Constructor
     * @param n Initialize the counter with a value of \c n.
     */
    Counter(boost::int64_t n = 0) :
            count_(n) {
    }

    virtual ~Counter() {
    }

    /**
     * @return the current value of the counter.
     */
    boost::int64_t getCount() const {
        return count_;
    }

    /**
     * @param n reset the counter to the value \c n.
     */
    void setCount(boost::int64_t n) {
        count_ = n;
    }

    /**
     * @param n increment the counter by \c n
     */
    void increment(boost::int64_t n = 1) {
        count_ += n;
    }

    /**
     * @param n decrement the counter by \c n
     */
    void decrement(boost::int64_t n = 1) {
        count_ -= n;
    }

    /**
     * Clears the counter, same as calling <code> setCount(0) </code>;
     */
    void clear() {
        setCount(0);
    }
private:
    boost::atomic<boost::int64_t> count_;
};

typedef boost::shared_ptr<Counter> CounterPtr;

} /* namespace core */
} /* namespace cppmetrics */
#endif /* COUNTER_H_ */
