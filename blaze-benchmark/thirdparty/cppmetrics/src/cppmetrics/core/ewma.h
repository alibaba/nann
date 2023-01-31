/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * ewma.h
 *
 *  Created on: Jun 4, 2014
 *      Author: vpoliboy
 */

#ifndef EWMA_H_
#define EWMA_H_

#include <boost/scoped_ptr.hpp>
#include <boost/chrono.hpp>
#include <boost/atomic.hpp>

namespace cppmetrics {
namespace core {
namespace internal {

/**
 * An exponentially-weighted moving average.
 * describe in detail  http://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
 * not thread-safe.
 */
class EWMA {
public:
    /**
     * Creates a new EWMA which is equivalent to the UNIX one minute load average and which expects
     * to be ticked every 5 seconds.
     * @return a one-minute EWMA
     */
    static EWMA oneMinuteEWMA() {
        return EWMA(M1_ALPHA, boost::chrono::seconds(INTERVAL_IN_SEC));
    }

    /**
     * Creates a new EWMA which is equivalent to the UNIX five minute load average and which expects
     * to be ticked every 5 seconds.
     * @return a five-minute EWMA
     */
    static EWMA fiveMinuteEWMA() {
        return EWMA(M5_ALPHA, boost::chrono::seconds(INTERVAL_IN_SEC));
    }

    /**
     * Creates a new EWMA which is equivalent to the UNIX fifteen minute load average and which expects
     * to be ticked every 5 seconds.
     * @return a five-minute EWMA
     */
    static EWMA fifteenMinuteEWMA() {
        return EWMA(M15_ALPHA, boost::chrono::seconds(INTERVAL_IN_SEC));
    }

    /**
     * Create a new EWMA with a specific smoothing constant.
     * @param alpha        the smoothing constant
     * @param interval     the expected tick interval
     */
    EWMA(double alpha, boost::chrono::nanoseconds interval);
    EWMA(const EWMA &other);
    ~EWMA();

    /**
     * Update the moving average with a new value.
     * @param n the new value
     */
    void update(boost::uint64_t n);

    /**
     * Mark the passage of time and decay the current rate accordingly.
     */
    void tick();

    /**
     * Returns the rate in the given units of time.
     * @param rate_unit the unit of time
     * @return the rate
     */
    double getRate(boost::chrono::nanoseconds rate_unit =
            boost::chrono::seconds(1)) const;
private:

    static const int INTERVAL_IN_SEC;
    static const int ONE_MINUTE;
    static const int FIVE_MINUTES;
    static const int FIFTEEN_MINUTES;
    static const double M1_ALPHA;
    static const double M5_ALPHA;
    static const double M15_ALPHA;

    boost::atomic<bool> initialized_;
    boost::atomic<double> ewma_;
    boost::atomic<boost::uint64_t> uncounted_;
    const double alpha_;
    const boost::uint64_t interval_nanos_;
};

} /* namespace internal */
} /* namespace core */
} /* namespace cppmetrics */

#endif /* EWMA_H_ */
