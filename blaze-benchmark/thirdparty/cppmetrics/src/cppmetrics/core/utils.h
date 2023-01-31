/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * utils.h
 *
 *  Created on: Jun 12, 2014
 *      Author: vpoliboy
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/time_facet.hpp>
#include "cppmetrics/core/types.h"

namespace cppmetrics {
namespace core {

inline boost::posix_time::time_duration get_duration_from_epoch() {
    boost::posix_time::ptime time_t_epoch(boost::gregorian::date(1970, 1, 1));
    boost::posix_time::ptime now =
            boost::posix_time::microsec_clock::local_time();
    return (now - time_t_epoch);
}

inline boost::uint64_t get_millis_from_epoch() {
    return get_duration_from_epoch().total_milliseconds();
}

inline boost::uint64_t get_seconds_from_epoch() {
    return get_duration_from_epoch().total_seconds();
}

inline std::string utc_timestamp(const std::locale& current_locale) {
    std::ostringstream ss;
    // assumes std::cout's locale has been set appropriately for the entire app
    boost::posix_time::time_facet* t_facet(new boost::posix_time::time_facet());
    t_facet->time_duration_format("%d-%M-%y %H:%M:%S%F %Q");
    ss.imbue(std::locale(current_locale, t_facet));
    ss << boost::posix_time::microsec_clock::universal_time();
    return ss.str();
}

}
}

#endif /* UTILS_H_ */
