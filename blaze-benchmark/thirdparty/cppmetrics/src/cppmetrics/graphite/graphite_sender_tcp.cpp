/*
 * Copyright 2000-2014 NeuStar, Inc. All rights reserved.
 * NeuStar, the Neustar logo and related names and logos are registered
 * trademarks, service marks or tradenames of NeuStar, Inc. All other
 * product names, company names, marks, logos and symbols may be trademarks
 * of their respective owners.
 */

/*
 * graphite_sender_tcp.cpp
 *
 *  Created on: Jun 16, 2014
 *      Author: vpoliboy
 */

#include <sstream>
#include <boost/lexical_cast.hpp>
#include "cppmetrics/graphite/graphite_sender_tcp.h"

namespace cppmetrics {
namespace graphite {

GraphiteSenderTCP::GraphiteSenderTCP(const std::string& host,
        boost::uint32_t port) :
                connected_(false),
                host_(host),
                port_(boost::lexical_cast<std::string>(port)) {
}

GraphiteSenderTCP::~GraphiteSenderTCP() {
}

void GraphiteSenderTCP::connect() {
    io_service_.reset(new boost::asio::io_service());

    boost::asio::ip::tcp::resolver resolver(*io_service_);
    boost::asio::ip::tcp::resolver::query query(host_, port_);
    boost::asio::ip::tcp::resolver::iterator iterator = resolver.resolve(query);

    socket_.reset(new boost::asio::ip::tcp::socket(*io_service_));
    boost::system::error_code ec;
    boost::asio::connect(*socket_, iterator, ec);
    connected_ = !ec;
    if (!connected_) {
        throw std::runtime_error("Connect() error, reason: " + ec.message());
    }
}

void GraphiteSenderTCP::send(const std::string& name,
        const std::string& value,
        boost::uint64_t timestamp) {
    if (!connected_) {
        throw std::runtime_error("Graphite server connection not established.");
    }
    std::ostringstream ostr;
    ostr << name << ' ' << value << ' ' << timestamp << std::endl;
    std::string graphite_str(ostr.str());
    boost::asio::write(*socket_,
            boost::asio::buffer(graphite_str, graphite_str.size()));
}

void GraphiteSenderTCP::close() {
    connected_ = false;
    socket_.reset();
    io_service_.reset();
}

} /* namespace graphite */
} /* namespace cppmetrics */
