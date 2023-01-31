/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * test_ewma.cpp
 *
 *  Created on: Jun 24, 2014
 *      Author: vpoliboy
 */

#include <gtest/gtest.h>
#include "cppmetrics/core/ewma.h"

namespace cppmetrics {
namespace core {
namespace internal {

namespace {
void elapse_minute(EWMA& ewma) {
    for (int i = 0; i < 12; i++) {
        ewma.tick();
    }
}
}

TEST(ewma, oneMinuteTestWithValueOf3) {

    EWMA ewma = EWMA::oneMinuteEWMA();
    ewma.update(3);
    ewma.tick();
    ASSERT_NEAR(0.6, ewma.getRate(boost::chrono::seconds(1)), 0.000001);
    elapse_minute(ewma);
    ASSERT_NEAR(0.22072766, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.08120117, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.02987224, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.01098938, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.00404277, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.00148725, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.00054713, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.00020128, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.00007405, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.00002724, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.00001002, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.00000369, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.00000136, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.00000050, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.00000018, ewma.getRate(boost::chrono::seconds(1)), 0.000001);
}

TEST(ewma, FiveMinuteTestWithValueOf3) {
    EWMA ewma = EWMA::fiveMinuteEWMA();
    ewma.update(3);
    ewma.tick();

    ASSERT_NEAR(0.6, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.49123845, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.40219203, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.32928698, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.26959738, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.22072766, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.18071653, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.14795818, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.12113791, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.09917933, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.08120117, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.06648190, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.05443077, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.04456415, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.03648604, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.02987224, ewma.getRate(boost::chrono::seconds(1)), 0.000001);
}

TEST(ewma, FifteenMinuteTestWithValueOf3) {
    EWMA ewma = EWMA::fifteenMinuteEWMA();
    ewma.update(3);
    ewma.tick();

    ASSERT_NEAR(0.6, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.56130419, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.52510399, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.49123845, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.45955700, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.42991879, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.40219203, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.37625345, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.35198773, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.32928698, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.30805027, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.28818318, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.26959738, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.25221023, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.23594443, ewma.getRate(boost::chrono::seconds(1)), 0.000001);

    elapse_minute(ewma);
    ASSERT_NEAR(0.22072766, ewma.getRate(boost::chrono::seconds(1)), 0.000001);
}

}
}
}

