/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * snapshot.h
 *
 *  Created on: Jun 5, 2014
 *      Author: vpoliboy
 */

#ifndef SNAPSHOT_H_
#define SNAPSHOT_H_

#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <vector>

namespace cppmetrics {
namespace core {

typedef std::vector<boost::int64_t> ValueVector;

/**
 * A statistical snapshot of a {@link Sample}.
 */
class Snapshot {
public:
    /**
     * Create a new {@link Snapshot} with the given values.
     * @param values    an unordered set of values in the reservoir
     */
    Snapshot(const ValueVector& values);
    ~Snapshot();

    /**
     * Returns the number of values in the snapshot.
     * @return the number of values
     */
    size_t size() const;

    /**
     * Returns the lowest value in the snapshot.
     * @return the lowest value
     */
    ValueVector::value_type getMin() const;

    /**
     * Returns the highest value in the snapshot.
     * @return the highest value
     */
    ValueVector::value_type getMax() const;

    /**
     * Returns the arithmetic mean of the values in the snapshot.
     * @return the arithmetic mean
     */
    double getMean() const;

    /**
     * Returns the standard deviation of the values in the snapshot.
     * @return the standard deviation value
     */
    double getStdDev() const;

    /**
     * Returns all the values in the snapshot.
     * @return All the values in the snapshot.
     */
    const ValueVector& getValues() const;

    /**
     * Returns the value at the given quantile.
     * @param quantile    a given quantile, in {@code [0..1]}
     * @return the value in the distribution at {@code quantile}
     */
    double getValue(double quantile) const;

    /**
     * Returns the median value in the distribution.
     * @return the median value
     */
    double getMedian() const;

    /**
     * Returns the value at the 75th percentile in the distribution.
     * @return the value at the 75th percentile
     */
    double get75thPercentile() const;

    /**
     * Returns the value at the 95th percentile in the distribution.
     * @return the value at the 95th percentile
     */
    double get95thPercentile() const;

    /**
     * Returns the value at the 98th percentile in the distribution.
     * @return the value at the 98th percentile
     */
    double get98thPercentile() const;

    /**
     * Returns the value at the 99th percentile in the distribution.
     * @return the value at the 99th percentile
     */
    double get99thPercentile() const;

    /**
     * Returns the value at the 999th percentile in the distribution.
     * @return the value at the 999th percentile
     */
    double get999thPercentile() const;
private:
    ValueVector values_;
};

typedef boost::shared_ptr<Snapshot> SnapshotPtr;

} /* namespace core */
} /* namespace cppmetrics */

#endif /* SNAPSHOT_H_ */
