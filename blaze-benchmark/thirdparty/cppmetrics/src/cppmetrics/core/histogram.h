/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * histogram.h
 *
 *  Created on: Jun 5, 2014
 *      Author: vpoliboy
 */

#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include <boost/cstdint.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/atomic.hpp>

#include "cppmetrics/core/metric.h"
#include "cppmetrics/core/sampling.h"
#include "cppmetrics/core/sample.h"

namespace cppmetrics {
namespace core {

/**
 * A metric which calculates the distribution of a value.
 * Thread-safe.
 */
class Histogram: public Metric, Sampling {
public:
    /**
     * Creates a new histogram based on the sample type.
     * @param sample_type the sample to use internally.
     * @see SamplingInterface for different types of samples.
     */
    Histogram(SampleType sample_type = kBiased);
    virtual ~Histogram();

    /**
     * @return the current snapshot based on the sample.
     */
    virtual SnapshotPtr getSnapshot() const;

    /**
     * Adds a recorded value.
     * @param value The length of the value.
     */
    void update(boost::int64_t value);

    /**
     * @return The number of values recorded until now.
     */
    boost::uint64_t getCount() const;

    /**
     * Clears the underlying sample.
     */
    void clear();

    /**< The Maximum sample size at any given time. */
    static const boost::uint64_t DEFAULT_SAMPLE_SIZE;
private:
    static const double DEFAULT_ALPHA;

    boost::scoped_ptr<Sample> sample_; /**< The underlying sample implementation. */
    boost::atomic<boost::uint64_t> count_; /**< The number of samples. */
};

typedef boost::shared_ptr<Histogram> HistogramPtr;

} /* namespace core */
} /* namespace cppmetrics */
#endif /* HISTOGRAM_H_ */
