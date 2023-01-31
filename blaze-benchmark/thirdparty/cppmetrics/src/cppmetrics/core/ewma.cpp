/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * ewma.cpp
 *
 *  Created on: Jun 4, 2014
 *      Author: vpoliboy
 */

#include <cmath>
#include "cppmetrics/core/ewma.h"

namespace cppmetrics {
namespace core {
namespace internal {

const int EWMA::INTERVAL_IN_SEC = 5;
const int EWMA::ONE_MINUTE = 1;
const int EWMA::FIVE_MINUTES = 5;
const int EWMA::FIFTEEN_MINUTES = 15;
// The following constants are calculated using the formulas used in computing linux load averages as described
// in http://www.perfdynamics.com/Papers/la1.pdf
const double EWMA::M1_ALPHA = 1
		- std::exp(static_cast<double>(-(EWMA::INTERVAL_IN_SEC)) / (60 * EWMA::ONE_MINUTE));
const double EWMA::M5_ALPHA = 1
		- std::exp(static_cast<double>(-(EWMA::INTERVAL_IN_SEC)) / (60 * EWMA::FIVE_MINUTES));
const double EWMA::M15_ALPHA = 1
		- std::exp(static_cast<double>(-(EWMA::INTERVAL_IN_SEC)) / (60 * EWMA::FIFTEEN_MINUTES));

EWMA::EWMA(double alpha, boost::chrono::nanoseconds interval) :
        uncounted_(0), alpha_(alpha), interval_nanos_(interval.count()) {
    initialized_ = false;
    ewma_ = 0.0;
}

EWMA::EWMA(const EWMA &other) :
                uncounted_(other.uncounted_.load()),
                alpha_(other.alpha_),
                interval_nanos_(other.interval_nanos_) {
    initialized_ = other.initialized_.load();
    ewma_ = other.ewma_.load();
}

EWMA::~EWMA() {
}

void EWMA::update(boost::uint64_t n) {
    uncounted_ += n;
}

// Uses the EWMA calculation described here:
// http://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
// EMAlatest = EMAprevious + alpha * (RATEtoday - EMAprevious)
void EWMA::tick() {
    const boost::uint64_t count = uncounted_.exchange(0);
    const double instant_rate = static_cast<double>(count) / interval_nanos_;
    if (initialized_) {
        // This does an atomic fetch and add.
        // TODO: Add a AtomicDouble class.
        double cur_ewma = ewma_;
        const double new_rate = cur_ewma + (alpha_ * (instant_rate - cur_ewma));
        ewma_.compare_exchange_strong(cur_ewma, new_rate);
    } else {
        ewma_ = instant_rate;
        initialized_ = true;
    }
}

double EWMA::getRate(boost::chrono::nanoseconds duration) const {
    return ewma_ * duration.count();
}

} /* namespace internal */
} /* namespace core */
} /* namespace cppmetrics */

