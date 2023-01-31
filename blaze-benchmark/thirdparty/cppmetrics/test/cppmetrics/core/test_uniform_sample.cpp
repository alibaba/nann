/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * test_uniform_sample.cpp
 *
 *  Created on: Jun 26, 2014
 *      Author: vpoliboy
 */

#include <gtest/gtest.h>
#include <boost/foreach.hpp>
#include "cppmetrics/core/uniform_sample.h"

namespace cppmetrics {
namespace core {

TEST(uniformsample, simpletest) {

    UniformSample uniform_sample(100);

    for (boost::uint64_t i = 0; i < 1000; i++) {
        uniform_sample.update(i);
    }
    SnapshotPtr snapshot = uniform_sample.getSnapshot();
    ASSERT_EQ((size_t )100, uniform_sample.size());
    ASSERT_EQ((size_t )100, snapshot->size());
    BOOST_FOREACH(ValueVector::value_type i, snapshot->getValues()) {
        ASSERT_LE(0, i);
        ASSERT_GT(1000, i);
    }
}

}
}
