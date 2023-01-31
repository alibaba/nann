/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * uniform_sample.cpp
 *
 *  Created on: Jun 5, 2014
 *      Author: vpoliboy
 */

#include "cppmetrics/core/utils.h"
#include "cppmetrics/core/uniform_sample.h"

namespace cppmetrics {
namespace core {

const boost::uint64_t UniformSample::DEFAULT_SAMPLE_SIZE = 1028;
UniformSample::UniformSample(boost::uint32_t reservoir_size) :
        reservoir_size_(reservoir_size), count_(0), values_(reservoir_size, 0) {
    rng_.seed(get_millis_from_epoch());
}

UniformSample::~UniformSample() {
}

void UniformSample::clear() {
    for (size_t i = 0; i < reservoir_size_; ++i) {
        values_[i] = 0;
    }
    count_ = 0;
}

boost::uint64_t UniformSample::size() const {
    boost::uint64_t size = values_.size();
    boost::uint64_t count = count_;
    return std::min(count, size);
}

boost::uint64_t UniformSample::getRandom(boost::uint64_t count) const {
    boost::random::uniform_int_distribution<> uniform(0, count - 1);
    return uniform(rng_);
}

void UniformSample::update(boost::int64_t value) {
    boost::uint64_t count = ++count_;
    boost::lock_guard<boost::mutex> lock(mutex_);
    size_t size = values_.size();
    if (count <= size) {
        values_[count - 1] = value;
    } else {
        boost::uint64_t rand = getRandom(count);
        if (rand < size) {
            values_[rand] = value;
        }
    }
}

SnapshotPtr UniformSample::getSnapshot() const {
    boost::lock_guard<boost::mutex> lock(mutex_);
    Int64Vector::const_iterator begin_itr(values_.begin());
    Int64Vector::const_iterator end_itr(values_.begin());
    std::advance(end_itr, size());
    return SnapshotPtr(new Snapshot(ValueVector(begin_itr, end_itr)));
}

} /* namespace core */
} /* namespace cppmetrics */
