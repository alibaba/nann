/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * metric_registry.cpp
 *
 *  Created on: Jun 9, 2014
 *      Author: vpoliboy
 */

#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/unordered_map.hpp>
#include "cppmetrics/core/metric_registry.h"

namespace cppmetrics {
namespace core {

class MetricRegistry::Impl {
public:
    Impl();
    ~Impl();

    bool addGauge(const std::string& name, GaugePtr metric);
    bool removeMetric(const std::string& name);

    CounterPtr counter(const std::string& name);
    HistogramPtr histogram(const std::string& name);
    MeterPtr meter(const std::string& name);
    TimerPtr timer(const std::string& name);

    CounterMap getCounters() const;
    HistogramMap getHistograms() const;
    MeteredMap getMeters() const;
    TimerMap getTimers() const;
    GaugeMap getGauges() const;

    size_t count() const;
private:
    // Old C++98 style enum.
    enum MetricType {
        GaugeType = 0,
        CounterType,
        HistogramType,
        MeterType,
        TimerType,
        TotalTypes
    };

    mutable boost::shared_mutex metrics_mutex_; /**< mutex that protects both MetricSets and metric_names. */
    // We should use a lock-free concurrent map implementation outside of boost.
    typedef boost::unordered_map<std::string, MetricPtr> MetricSet;
    MetricSet metric_set_[TotalTypes];
    typedef std::set<std::string> StringSet;
    StringSet metric_names_;

    template<typename MetricClass>
    bool isInstanceOf(const MetricPtr& metric_ptr) const;

    bool addMetric(MetricSet& metric_set,
            const std::string& name,
            MetricPtr metric);

    MetricPtr buildMetric(MetricType metric_type) const;
    MetricPtr getOrAdd(MetricType metric_type, const std::string& name);

    template<typename MetricClass>
    std::map<std::string, boost::shared_ptr<MetricClass> > getMetrics(const MetricSet& metric_set) const;
};

MetricRegistry::Impl::Impl() {

}

MetricRegistry::Impl::~Impl() {

}

size_t MetricRegistry::Impl::count() const {
    boost::shared_lock<boost::shared_mutex> read_lock(metrics_mutex_);
    return metric_names_.size();
}

// RTTI is a performance overhead, should probably replace it in future.
template<typename MetricClass>
bool MetricRegistry::Impl::isInstanceOf(const MetricPtr& metric_ptr) const {
    boost::shared_ptr<MetricClass> stored_metric(
            boost::dynamic_pointer_cast<MetricClass>(metric_ptr));
    return (stored_metric.get() != NULL);
}

MetricPtr MetricRegistry::Impl::buildMetric(MetricType metric_type) const {
    MetricPtr metric_ptr;
    switch (metric_type) {
    case CounterType:
        return boost::make_shared<Counter>();
    case HistogramType:
        return boost::make_shared<Histogram>();
    case MeterType:
        return boost::make_shared<Meter>();
    case TimerType:
        return boost::make_shared<Timer>();
    default:
        throw std::invalid_argument("Unknown or invalid metric type.");
    };
}

bool MetricRegistry::Impl::addMetric(MetricSet& metric_set,
        const std::string& name,
        MetricPtr new_metric) {
    StringSet::iterator s_itt(metric_names_.find(name));
    if (s_itt == metric_names_.end()) {
        metric_names_.insert(name);
        return metric_set.insert(std::make_pair(name, new_metric)).second;
    }
    throw std::invalid_argument(
            name + " already exists as a different metric.");
}

MetricPtr MetricRegistry::Impl::getOrAdd(MetricType metric_type,
        const std::string& name) {
    boost::unique_lock<boost::shared_mutex> wlock(metrics_mutex_);
    MetricSet& metric_set(metric_set_[metric_type]);
    MetricSet::iterator itt(metric_set.find(name));
    if (itt != metric_set.end()) {
        return itt->second;
    } else {
        MetricPtr new_metric(buildMetric(metric_type));
        addMetric(metric_set, name, new_metric);
        return new_metric;
    }
}

bool MetricRegistry::Impl::addGauge(const std::string& name, GaugePtr gauge) {
    boost::unique_lock<boost::shared_mutex> wlock(metrics_mutex_);
    return addMetric(metric_set_[GaugeType], name, gauge);
}

CounterPtr MetricRegistry::Impl::counter(const std::string& name) {
    MetricPtr metric_ptr(getOrAdd(CounterType, name));
    return boost::static_pointer_cast<Counter>(metric_ptr);
}

HistogramPtr MetricRegistry::Impl::histogram(const std::string& name) {
    MetricPtr metric_ptr(getOrAdd(HistogramType, name));
    return boost::static_pointer_cast<Histogram>(metric_ptr);
}

MeterPtr MetricRegistry::Impl::meter(const std::string& name) {
    MetricPtr metric_ptr(getOrAdd(MeterType, name));
    return boost::static_pointer_cast<Meter>(metric_ptr);
}

TimerPtr MetricRegistry::Impl::timer(const std::string& name) {
    MetricPtr metric_ptr(getOrAdd(TimerType, name));
    return boost::static_pointer_cast<Timer>(metric_ptr);
}

template<typename MetricClass>
std::map<std::string, boost::shared_ptr<MetricClass> >
MetricRegistry::Impl::getMetrics(const MetricSet& metric_set) const {
    std::map<std::string, boost::shared_ptr<MetricClass> > ret_set;
    boost::shared_lock<boost::shared_mutex> rlock(metrics_mutex_);
    BOOST_FOREACH (const MetricSet::value_type& kv, metric_set) {
        ret_set[kv.first] = boost::static_pointer_cast<MetricClass>(kv.second);
    }
    return ret_set;
}

CounterMap MetricRegistry::Impl::getCounters() const {
    return getMetrics<Counter>(metric_set_[CounterType]);
}

HistogramMap MetricRegistry::Impl::getHistograms() const {
    return getMetrics<Histogram>(metric_set_[HistogramType]);
}

MeteredMap MetricRegistry::Impl::getMeters() const {
    return getMetrics<Metered>(metric_set_[MeterType]);
}

TimerMap MetricRegistry::Impl::getTimers() const {
    return getMetrics<Timer>(metric_set_[TimerType]);
}

GaugeMap MetricRegistry::Impl::getGauges() const {
    return getMetrics<Gauge>(metric_set_[GaugeType]);
}

bool MetricRegistry::Impl::removeMetric(const std::string& name) {
    boost::unique_lock<boost::shared_mutex> wlock(metrics_mutex_);
    StringSet::iterator s_itt(metric_names_.find(name));
    if (s_itt != metric_names_.end()) {
        for (size_t i = 0; i < TotalTypes; ++i) {
            if (metric_set_[i].erase(name) > 0) {
                break;
            }
        }
        metric_names_.erase(s_itt);
        return true;
    }
    return false;
}

// <=================Implementation end============>

MetricRegistryPtr MetricRegistry::DEFAULT_REGISTRY() {
    static MetricRegistryPtr g_metric_registry(new MetricRegistry());
    return g_metric_registry;
}

MetricRegistry::MetricRegistry() :
        impl_(new MetricRegistry::Impl()) {
}

MetricRegistry::~MetricRegistry() {
}

CounterPtr MetricRegistry::counter(const std::string& name) {
    return impl_->counter(name);
}

HistogramPtr MetricRegistry::histogram(const std::string& name) {
    return impl_->histogram(name);
}

MeterPtr MetricRegistry::meter(const std::string& name) {
    return impl_->meter(name);
}

TimerPtr MetricRegistry::timer(const std::string& name) {
    return impl_->timer(name);
}

CounterMap MetricRegistry::getCounters() const {
    return impl_->getCounters();
}

HistogramMap MetricRegistry::getHistograms() const {
    return impl_->getHistograms();
}

MeteredMap MetricRegistry::getMeters() const {
    return impl_->getMeters();
}

TimerMap MetricRegistry::getTimers() const {
    return impl_->getTimers();
}

GaugeMap MetricRegistry::getGauges() const {
    return impl_->getGauges();
}

size_t MetricRegistry::count() const {
    return impl_->count();
}

bool MetricRegistry::addGauge(const std::string& name, GaugePtr metric) {
    return impl_->addGauge(name, metric);
}

bool MetricRegistry::removeMetric(const std::string& name) {
    return impl_->removeMetric(name);
}

} /* namespace core */
} /* namespace cppmetrics */
