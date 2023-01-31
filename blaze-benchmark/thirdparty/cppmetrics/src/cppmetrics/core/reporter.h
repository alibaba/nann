/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * reporter.h
 *
 *  Created on: Jun 10, 2014
 *      Author: vpoliboy
 */

#ifndef REPORTER_H_
#define REPORTER_H_

namespace cppmetrics {
namespace core {

/**
 * The interface for all the reporter sub classes.
 */
class Reporter {
public:
    virtual ~Reporter() {
    }

    /**
     * reports the metrics.
     */
    virtual void report() = 0;
};

} /* namespace core */
} /* namespace cppmetrics */
#endif /* REPORTER_H_ */
