/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * metric.h
 *
 *  Created on: Jun 4, 2014
 *      Author: vpoliboy
 */

#ifndef METRIC_H_
#define METRIC_H_

#include <boost/shared_ptr.hpp>

namespace cppmetrics {
namespace core {

/**
 * The base class for all metrics types.
 */
class Metric {
public:
    virtual ~Metric() = 0;
};

inline Metric::~Metric() {
}

typedef boost::shared_ptr<Metric> MetricPtr;

} /* namespace core */
} /* namespace cppmetrics */
#endif /* METRIC_H_ */
