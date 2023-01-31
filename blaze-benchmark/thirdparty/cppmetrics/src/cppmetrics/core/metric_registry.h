/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * metric_registry.h
 *
 *  Created on: Jun 9, 2014
 *      Author: vpoliboy
 */

#ifndef METRIC_REGISTRY_H_
#define METRIC_REGISTRY_H_

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <string>
#include <map>
#include "cppmetrics/core/counter.h"
#include "cppmetrics/core/gauge.h"
#include "cppmetrics/core/histogram.h"
#include "cppmetrics/core/meter.h"
#include "cppmetrics/core/timer.h"

namespace cppmetrics {
namespace core {

typedef std::map<std::string, CounterPtr> CounterMap;
typedef std::map<std::string, HistogramPtr> HistogramMap;
typedef std::map<std::string, MeteredPtr> MeteredMap;
typedef std::map<std::string, TimerPtr> TimerMap;
typedef std::map<std::string, GaugePtr> GaugeMap;

class MetricRegistry;
typedef boost::shared_ptr<MetricRegistry> MetricRegistryPtr;
/**
 * The thread-safe registry class for all metrics.
 */
class MetricRegistry: boost::noncopyable {
public:

    /**
     * Singleton factory method for the Metric registry.
     * @return The default singleton metric registry
     */
    static MetricRegistryPtr DEFAULT_REGISTRY();

    /**
     * Creates a new registry.
     */
    MetricRegistry();

    ~MetricRegistry();

    /**
     * Adds a gauge with the given name to the registry.
     * @param name The name of the gauge metric.
     * @param metric A subclass object of the Gauge.
     * @return True on creation, false if the gauge is already present.
     * @throws std::invalid_argument exception if a metric of different type with the same name is already present.
     */
    bool addGauge(const std::string& name, GaugePtr metric);

    /**
     * Removes the metric with the given name from the registry.
     * @param name The name of the metric
     * @return True on success, false if the metric with the name is not present.
     */
    bool removeMetric(const std::string& name);

    /**
     * Gets a {@link Counter} from the registry with the given name if present otherwise creates and adds a new
     *  {@link Counter} and returns the newly added one.
     * @param name The name of the Counter.
     * @return shared_ptr to the Counter object.
     * @throws std::invalid_argument if a metric with the same name but different type exists in the registry.
     */
    CounterPtr counter(const std::string& name);

    /**
     * Gets a {@link Histogram} from the registry with the given name if present otherwise creates and adds a new
     *  {@link Histogram} and returns the newly added one.
     * @param name The name of the Histogram.
     * @return shared_ptr to the Histogram ojbect.
     * @throws std::invalid_argument if a metric with the same name but different type exists in the registry.
     */
    HistogramPtr histogram(const std::string& name);

    /**
     * Gets a {@link Meter} from the registry with the given name if present otherwise creates and adds a new
     *  {@link Meter} and returns the newly added one.
     * @param name The name of the Meter.
     * @return shared_ptr to the Meter ojbect.
     * @throws std::invalid_argument if a metric with the same name but different type exists in the registry.
     */
    MeterPtr meter(const std::string& name);

    /**
     * Gets a {@link Timer} from the registry with the given name if present otherwise creates and adds a new
     *  {@link Timer} and returns the newly added one.
     * @param name The name of the Timer.
     * @return shared_ptr to the Timer ojbect.
     * @throws std::invalid_argument if a metric with the same name but different type exists in the registry.
     */
    TimerPtr timer(const std::string& name);

    /**
     * Returns all the counters and their names currently in the registry.
     * @return all the counters in the registry.
     */
    CounterMap getCounters() const;

    /**
     * Returns all the histograms and their names currently in the registry.
     * @return all the histograms in the registry.
     */
    HistogramMap getHistograms() const;

    /**
     * Returns all the meters and their names currently in the registry.
     * @return all the meters in the registry.
     */
    MeteredMap getMeters() const;

    /**
     * Returns all the timers and their names currently in the registry.
     * @return all the timers in the registry.
     */
    TimerMap getTimers() const;

    /**
     * Returns all the gauges and their names currently in the registry.
     * @return all the gauges in the registry.
     */
    GaugeMap getGauges() const;

    /**
     * Gets the total number of metrics in the registry.
     * @return the total metrics count.
     */
    size_t count() const;

private:
    class Impl;
    boost::scoped_ptr<Impl> impl_; /**< The pimpl pointer */
};

} /* namespace core */
} /* namespace cppmetrics */
#endif /* METRIC_REGISTRY_H_ */
