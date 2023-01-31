/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * snapshot.cpp
 *
 *  Created on: Jun 5, 2014
 *      Author: vpoliboy
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <boost/foreach.hpp>
#include "cppmetrics/core/snapshot.h"

namespace cppmetrics {
namespace core {

static const double MEDIAN_Q = 0.5;
static const double P75_Q = 0.75;
static const double P95_Q = 0.95;
static const double P98_Q = 0.98;
static const double P99_Q = 0.99;
static const double P999_Q = 0.999;

Snapshot::Snapshot(const ValueVector& values) :
        values_(values) {
    std::sort(values_.begin(), values_.end());
}

Snapshot::~Snapshot() {
}

std::size_t Snapshot::size() const {
    return values_.size();
}

double Snapshot::getValue(double quantile) const {
    if (quantile < 0.0 || quantile > 1.0) {
        throw std::invalid_argument("quantile is not in [0..1]");
    }

    if (values_.empty()) {
        return 0.0;
    }

    const double pos = quantile * (values_.size() + 1);

    if (pos < 1) {
        return values_.front();
    }

    if (pos >= values_.size()) {
        return values_.back();
    }

    const size_t pos_index = static_cast<size_t>(pos);
    double lower = values_[pos_index - 1];
    double upper = values_[pos_index];
    return lower + (pos - std::floor(pos)) * (upper - lower);
}

double Snapshot::getMedian() const {
    return getValue(MEDIAN_Q);
}

double Snapshot::get75thPercentile() const {
    return getValue(P75_Q);
}

double Snapshot::get95thPercentile() const {
    return getValue(P95_Q);
}

double Snapshot::get98thPercentile() const {
    return getValue(P98_Q);
}

double Snapshot::get99thPercentile() const {
    return getValue(P99_Q);
}

double Snapshot::get999thPercentile() const {
    return getValue(P999_Q);
}

ValueVector::value_type Snapshot::getMin() const {
    return (values_.empty() ? 0.0 : values_.front());
}

ValueVector::value_type Snapshot::getMax() const {
    return (values_.empty() ? 0.0 : values_.back());
}

double Snapshot::getMean() const {
    if (values_.empty()) {
        return 0.0;
    }

    ValueVector::value_type mean(0);
    BOOST_FOREACH(ValueVector::value_type d, values_) {
        mean += d;
    }
    return static_cast<double>(mean) / values_.size();
}

double Snapshot::getStdDev() const {
    const size_t values_size(values_.size());
    if (values_size <= 1) {
        return 0.0;
    }

    double mean_value = getMean();
    double sum = 0;

    BOOST_FOREACH(ValueVector::value_type value, values_) {
        double diff = static_cast<double>(value) - mean_value;
        sum += diff * diff;
    }

    double variance = sum / (values_size - 1);
    return std::sqrt(variance);
}

const ValueVector& Snapshot::getValues() const {
    return values_;
}

} /* namespace core */
} /* namespace cppmetrics */
