/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * sampling.h
 *
 *  Created on: Jun 5, 2014
 *      Author: vpoliboy
 */

#ifndef SAMPLING_H_
#define SAMPLING_H_

#include "cppmetrics/core/snapshot.h"

namespace cppmetrics {
namespace core {

/**
 * The interface for all classes that sample values.
 */
class Sampling {
public:
    enum SampleType {
        kUniform, kBiased
    };
    virtual ~Sampling() {
    }

    /**
     * Returns the snapshot of values in the sample.
     * @return the snapshot of values in the sample.
     */
    virtual SnapshotPtr getSnapshot() const = 0;
};

} /* namespace core */
} /* namespace cppmetrics */
#endif /* SAMPLING_H_ */
