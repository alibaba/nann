/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * uniform_sample.h
 *
 *  Created on: Jun 5, 2014
 *      Author: vpoliboy
 */

#ifndef UNIFORM_SAMPLE_H_
#define UNIFORM_SAMPLE_H_

#include <vector>
#include <iterator>
#include <boost/atomic.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include "cppmetrics/core/sample.h"

namespace cppmetrics {
namespace core {

/**
 * A random sampling reservoir of a stream of {@code long}s. Uses Vitter's Algorithm R to produce a
 * statistically representative sample.
 */
class UniformSample: public Sample {
public:

    /**
     * Creates a new {@link UniformReservoir}.
     * @param size the number of samples to keep in the sampling reservoir
     */
    UniformSample(boost::uint32_t reservoirSize = DEFAULT_SAMPLE_SIZE);
    virtual ~UniformSample();

    /**
     * Clears the values in the sample.
     */
    virtual void clear();

    /**
     * Returns the number of values recorded.
     * @return the number of values recorded
     */
    virtual boost::uint64_t size() const;

    /**
     * Adds a new recorded value to the sample.
     * @param value a new recorded value
     */
    virtual void update(boost::int64_t value);

    /**
     * Returns a snapshot of the sample's values.
     * @return a snapshot of the sample's values
     */
    virtual SnapshotPtr getSnapshot() const;

    /**< The Maximum sample size at any given time. */
    static const boost::uint64_t DEFAULT_SAMPLE_SIZE;
private:
    boost::uint64_t getRandom(boost::uint64_t count) const;
    const boost::uint64_t reservoir_size_;
    boost::atomic<boost::uint64_t> count_;
    typedef std::vector<boost::int64_t> Int64Vector;
    Int64Vector values_;
    mutable boost::mt11213b rng_;
    mutable boost::mutex mutex_;
};

} /* namespace core */
} /* namespace cppmetrics */
#endif /* UNIFORM_SAMPLE_H_ */
