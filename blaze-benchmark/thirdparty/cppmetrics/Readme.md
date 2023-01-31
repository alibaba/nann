##Background

cppmetrics is a C++ port of the [DropWizard metrics!](https://dropwizard.github.io/metrics/3.1.0/).
The library implements the standard metrics primitives like Gauge, Counter, Histogram, Meter and Timer and the provides the reporter
implementations like the ConsoleReporter, GraphiteRepoter out of the box.
Its written in C++98 to make the integration into existing pre-C++11 codebases easier and should be portable across different 
platforms but being used only in linux environment.

[![Build Status](https://travis-ci.org/ultradns/cppmetrics.png)](https://travis-ci.org/ultradns/cppmetrics)

## Build dependencies
- cmake (>= 2.6.5)
- boost libraries (>= 1.53.0)
- google logging framework (>= 0.3.1)
- gtest (>= 1.6.0, dependency for the unit tests only.)

## How to build

```
# It is recommended to create the build directory in the parent directory of cppmetrics source as opposed to creating in the cppmetrics directory.
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_INSTALL_PREFIX=dist -DBOOST_DIR=<BOOST_BINARY_DISTRO> -DGLOG_DIR=<GLOG_BINARY_DISTRO> -DGTEST_DIR=<GTEST_BINARY_DISTRO> ../cppmetrics/
make gtest
make package
```

The above process produces a tar file of include files and a static library and should be used to statically link in your existing application. The shared library option is
not turned off by default but can be turned on easily if required.

##Sample code snippet

####Using a Histogram or a timer or a meter..
``` cpp
#include <cppmetrics/cppmetrics.h>
...

bool QueryHandler::doProcess(const Query& query) {
    cppmetrics::core::MetricRegistryPtr registry(
            cppmetrics::core::MetricRegistry::DEFAULT_REGISTRY());

    // More initialization.

    cppmetrics::core::CounterPtr query_counter(registry->counter("get_requests"));
    query_counter->increment();
    // More  processing
    {
       cppmetrics::core::TimerContextPtr timer(
                metrics->timer("query_process")->timerContextPtr());
       // Do some computation or IO.
       // timer stats will be updated in the registry at the end of the scope.                
    }
}
```

####Creating the default metrics registry and a graphite reporter that pushes the data to graphite server.

```cpp
#include <boost/noncopyable.hpp>
#include <cppmetrics/cppmetrics.h>
#include <glog/logging.h>

namespace sample {

class GraphiteReporterOptions {
public:
    std::string host_;                  ///<  The graphite server.
    boost::uint32_t port_;              ///<  The graphite port.
    std::string prefix_;                ///<  The prefix to the graphite.
    boost::uint32_t interval_in_secs_;  ///<  The reporting period in secs.
};

/*
 *  Helper class that sets up the default registry and the graphite reporter.
 */
class Controller : boost::noncopyable
{
public:
    cppmetrics::core::MetricRegistryPtr getRegistry() {
        return core::MetricRegistry::DEFAULT_REGISTRY();
    }
    
    void configureAndStartGraphiteReporter(const GraphiteReporterOptions& graphite_options) {
        if (!graphite_reporter_) {
            const std::string& graphite_host(graphite_options.host_);

            boost::uint32_t graphite_port(graphite_options.port_);
            graphite::GraphiteSenderPtr graphite_sender(
                new graphite::GraphiteSenderTCP(graphite_host, graphite_port));

            graphite_reporter_.reset(
                new graphite::GraphiteReporter(getRegistry(), graphite_sender,
                        graphite_options.prefix_));
            graphite_reporter_->start(boost::chrono::seconds(graphite_options.interval_in_secs_));
        } else {
            LOG(ERROR) << "Graphite reporter already configured.";
        }
    }
private:
    boost::scoped_ptr<cppmetrics::graphite::GraphiteReporter> graphite_reporter_;
};

}
```

###TODO
- Currently the Timer and Meter resolutions are in millis and per-minute respectively, make this configurable.
- Provide more reporters out of the box.


