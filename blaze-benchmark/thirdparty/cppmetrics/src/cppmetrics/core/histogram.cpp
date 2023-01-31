/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * histogram.cpp
 *
 *  Created on: Jun 5, 2014
 *      Author: vpoliboy
 */

#include "cppmetrics/core/uniform_sample.h"
#include "cppmetrics/core/exp_decay_sample.h"
#include "cppmetrics/core/histogram.h"

namespace cppmetrics {
namespace core {

Histogram::Histogram(SampleType sample_type) {
    if (sample_type == kUniform) {
        sample_.reset(new UniformSample());
    } else if (sample_type == kBiased) {
        sample_.reset(new ExpDecaySample());
    } else {
        throw std::invalid_argument("invalid sample_type.");
    }
    clear();
}

Histogram::~Histogram() {
}

void Histogram::clear() {
    count_ = 0;
    sample_->clear();
}

boost::uint64_t Histogram::getCount() const {
    return count_;
}

SnapshotPtr Histogram::getSnapshot() const {
    return sample_->getSnapshot();
}

void Histogram::update(boost::int64_t value) {
    ++count_;
    sample_->update(value);
}

} /* namespace core */
} /* namespace cppmetrics */
