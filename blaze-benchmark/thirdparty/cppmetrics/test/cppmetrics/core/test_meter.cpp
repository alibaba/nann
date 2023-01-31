/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * test_meter.cpp
 *
 *  Created on: Jun 24, 2014
 *      Author: vpoliboy
 */


#include <gtest/gtest.h>
#include "cppmetrics/core/meter.h"

namespace cppmetrics {
namespace core {

TEST(meter, functionaltest) {

	Meter meter;

	ASSERT_EQ((size_t)0, meter.getCount());
	ASSERT_NEAR(0.0, meter.getMeanRate(), 0.001);
	ASSERT_NEAR(0.0, meter.getOneMinuteRate(), 0.001);
	ASSERT_NEAR(0.0, meter.getFiveMinuteRate(), 0.001);
	ASSERT_NEAR(0.0, meter.getFifteenMinuteRate(), 0.001);
/*
 	// TODO: Have to use gmock here.
	meter.mark();
	meter.mark(2);

	ASSERT_NEAR(0.3, meter.getMeanRate(), 0.001);
	ASSERT_NEAR(0.1840, meter.getOneMinuteRate(), 0.001);
	ASSERT_NEAR(0.1966, meter.getFiveMinuteRate(), 0.001);
	ASSERT_NEAR(0.1988, meter.getFifteenMinuteRate(), 0.001);
	*/
}

}
}

