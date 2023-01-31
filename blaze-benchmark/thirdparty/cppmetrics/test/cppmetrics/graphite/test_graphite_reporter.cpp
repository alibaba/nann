/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * test_graphite_reporter.cpp
 *
 *  Created on: Jun 26, 2014
 *      Author: vpoliboy
 */

#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include "cppmetrics/graphite/graphite_reporter.h"

namespace cppmetrics {
namespace graphite {

static const std::string PREFIX("Prefix");
static const std::string GAUGE_NAME("Gauge");
static const boost::int64_t GAUGE_VALUE(100);


static const std::string COUNTER_NAME("Counter");
static const boost::uint64_t COUNTER_VALUE(100);


namespace {


class FakeGauge : public core::Gauge {
public:
	virtual ~FakeGauge() {}
    virtual boost::int64_t getValue() {
    	return GAUGE_VALUE;
    }
};


// TODO: use gmock here instead.
class FakeGraphiteSender : public GraphiteSender {
public:
	enum MetricType {Counter, Gauge, Histogram, Meter, Timer};
	FakeGraphiteSender(MetricType metric_type) : metric_type_(metric_type) {}

	virtual ~FakeGraphiteSender() {
		verifyMethodCallOrder();
	};

	virtual void connect() {
		method_called_[Connect] = true;
	}

	virtual void send(const std::string& name, const std::string& value,
			boost::uint64_t timestamp) {
		ASSERT_TRUE(method_called_[Connect]);
		method_called_[Send] = true;
		switch (metric_type_) {
		case Counter:
			sendCounter(name, value, timestamp);
			break;
		case Gauge:
			sendGauge(name, value, timestamp);
			break;
		case Histogram:
			sendHistogram(name, value, timestamp);
			break;
		case Meter:
			sendMeter(name, value, timestamp);
			break;
		case Timer:
			sendTimer(name, value, timestamp);
			break;
		default:
			ASSERT_EQ(2, 1);
		}
	}

	virtual void close() {
		ASSERT_TRUE(method_called_[Connect]);
		ASSERT_TRUE(method_called_[Send]);
		method_called_[Close] = true;
	}

private:
	enum METHOD { Connect = 0, Send, Close, TotalMethods};
	bool method_called_[TotalMethods];

	MetricType metric_type_;

	void verifyMethodCallOrder() {
		ASSERT_TRUE(method_called_[Connect]);
		ASSERT_TRUE(method_called_[Send]);
		ASSERT_TRUE(method_called_[Close]);
	}

	void sendGauge(const std::string& name, const std::string& actual_value,
			boost::uint64_t timestamp) {
		std::string expected_value(boost::lexical_cast<std::string>(GAUGE_VALUE));
		ASSERT_STREQ(std::string(PREFIX + '.' + GAUGE_NAME).c_str(), name.c_str());
		ASSERT_STREQ(expected_value.c_str(), actual_value.c_str());
	}

	void sendCounter(const std::string& name, const std::string& actual_value,
			boost::uint64_t timestamp) {
		std::string expected_value(boost::lexical_cast<std::string>(COUNTER_VALUE));
		ASSERT_STREQ(std::string(PREFIX + '.' + COUNTER_NAME + ".count").c_str(), name.c_str());
		ASSERT_STREQ(expected_value.c_str(), actual_value.c_str());
	}

	void sendHistogram(const std::string& name, const std::string& value,
			boost::uint64_t timestamp) {

	}

	void sendMeter(const std::string& name, const std::string& value,
			boost::uint64_t timestamp) {

	}

	void sendTimer(const std::string& name, const std::string& value,
			boost::uint64_t timestamp) {

	}
};

}

TEST(graphitereporter, gaugetest) {
	core::MetricRegistryPtr metric_registry(new core::MetricRegistry());
	GraphiteSenderPtr graphite_sender(new FakeGraphiteSender(FakeGraphiteSender::Gauge));

	core::GaugePtr gauge_ptr(new FakeGauge());
	metric_registry->addGauge(GAUGE_NAME, gauge_ptr);

	GraphiteReporter graphite_reporter(metric_registry,
			graphite_sender, PREFIX,
			boost::chrono::seconds(1));

	graphite_reporter.start(boost::chrono::milliseconds(100));
	boost::this_thread::sleep(boost::posix_time::milliseconds(150));
	graphite_reporter.stop();
}

TEST(graphitereporter, countertest) {
	core::MetricRegistryPtr metric_registry(new core::MetricRegistry());
	GraphiteSenderPtr graphite_sender(new FakeGraphiteSender(FakeGraphiteSender::Counter));


	core::CounterPtr counter_ptr(metric_registry->counter(COUNTER_NAME));
	counter_ptr->increment(COUNTER_VALUE);

	GraphiteReporter graphite_reporter(metric_registry,
			graphite_sender, PREFIX,
			boost::chrono::seconds(1));

	graphite_reporter.start(boost::chrono::milliseconds(100));
	boost::this_thread::sleep(boost::posix_time::milliseconds(150));
	graphite_reporter.stop();
}

}
}


