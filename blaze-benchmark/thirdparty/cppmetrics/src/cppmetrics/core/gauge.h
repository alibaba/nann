/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * gauge.h
 *
 *  Created on: Jun 9, 2014
 *      Author: vpoliboy
 */

#ifndef GAUGE_H_
#define GAUGE_H_

#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include "cppmetrics/core/metric.h"

namespace cppmetrics {
namespace core {

/**
 * A gauge metric is an instantaneous reading of a particular value. Used typically
 * to instrument a queue size, backlog etc.
 *
 */
class Gauge: public Metric {
public:
    virtual ~Gauge() {
    }

    /**
     * @return the current value of the guage.
     */
    virtual boost::int64_t getValue() = 0;
};

typedef boost::shared_ptr<Gauge> GaugePtr;

} /* namespace core */
} /* namespace cppmetrics */
#endif /* GAUGE_H_ */
