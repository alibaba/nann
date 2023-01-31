/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * sample.h
 *
 *  Created on: Jun 5, 2014
 *      Author: vpoliboy
 */

#ifndef SAMPLE_H_
#define SAMPLE_H_

#include "cppmetrics/core/snapshot.h"

namespace cppmetrics {
namespace core {

/**
 * A statistically representative sample of a data stream.
 */
class Sample {
public:
    virtual ~Sample() {
    }

    /**
     * Clears the values in the sample.
     */
    virtual void clear() = 0;

    /**
     * Returns the number of values recorded.
     * @return the number of values recorded
     */
    virtual boost::uint64_t size() const = 0;

    /**
     * Adds a new recorded value to the sample.
     * @param value a new recorded value
     */
    virtual void update(boost::int64_t value) = 0;

    /**
     * Returns a snapshot of the sample's values.
     * @return a snapshot of the sample's values
     */
    virtual SnapshotPtr getSnapshot() const = 0;
};

} /* namespace core */
} /* namespace cppmetrics */
#endif /* SAMPLE_H_ */
