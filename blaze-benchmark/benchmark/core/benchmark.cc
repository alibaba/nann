#include <unistd.h>

#include "benchmark/common/proto_configure.h"
#include "benchmark/core/metrics.h"
#include "benchmark/core/model.h"
#include "benchmark/core/predict_request_consumer.h"
#include "benchmark/core/predict_request_producer.h"
#include "benchmark/proto/bench_conf.pb.h"

using benchmark::BenchmarkConfig;
using benchmark::Metrics;
using benchmark::Model;
using benchmark::ModelReloader;
using benchmark::ModelSelector;
using benchmark::PredictRequestConsumer;
using benchmark::PredictRequestProducer;
using benchmark::PredictRequestQueue;
using benchmark::ProtoConfigure;

void ShowHelp(const char* exe) {
  std::cerr << "Usage: " << exe << " benchmark_conf" << std::endl;
}
bool FileExists(const std::string& conf_dir, std::string* path);

int main(int argc, char* argv[]) {
  const char* bench_conf_path = argc > 1 ? argv[1] : nullptr;
  if (bench_conf_path == nullptr) {
    ShowHelp(argv[0]);
    return 1;
  }

  // Parse benchmark config
  ProtoConfigure proto_conf;
  auto status = proto_conf.Init("benchmark.BenchmarkConfig", bench_conf_path);
  if (status != ProtoConfigure::kOK) {
    ShowHelp(argv[0]);
    return 1;
  }
  auto benchmark_config =
      dynamic_cast<BenchmarkConfig*>(proto_conf.mutable_config());

  std::string conf_dir = bench_conf_path;
  size_t pos = conf_dir.find_last_of("/");
  if (pos != std::string::npos) {
    conf_dir = conf_dir.substr(0, pos + 1);
  } else {
    conf_dir = "./";
  }

  // Init all models
  std::unique_ptr<ModelSelector> model_selector(new ModelSelector());
  for (auto bench_model_conf : *benchmark_config->mutable_bench_model_config()) {
    auto frozen_graph_file = bench_model_conf.mutable_frozen_graph();
    // Check and concat folder name
    if (!FileExists(conf_dir, frozen_graph_file)) {
      LOG(ERROR) << "Invalid frozen_graph path: " << *frozen_graph_file;
      ShowHelp(argv[0]);
      return 1;
    }
    auto runmeta_files = bench_model_conf.mutable_runmeta();
    for (auto& file : *runmeta_files) {
      if (!FileExists(conf_dir, &file)) {
        LOG(ERROR) << "Invalid runmeta path: " << file;
        ShowHelp(argv[0]);
        return 1;
      }
    }
    auto meta_graph_file = bench_model_conf.mutable_meta_graph();
    if (!FileExists(conf_dir, meta_graph_file)) {
      LOG(ERROR) << "Invalid runmeta path: " << *meta_graph_file;
      ShowHelp(argv[0]);
      return 1;
    }
    auto config_proto_file = bench_model_conf.mutable_config_proto();
    if (!FileExists(conf_dir, config_proto_file)) {
      LOG(ERROR) << "Invalid config proto path: " << *config_proto_file
                 << ", use default config proto.";
    }
    auto run_options_file = bench_model_conf.mutable_run_options();
    if (!FileExists(conf_dir, run_options_file)) {
      LOG(ERROR) << "Invalid run options path: " << *run_options_file
                 << ", use default run options.";
    }
    if (!model_selector->InitModel(bench_model_conf)) {
      LOG(ERROR) << "Init model " << bench_model_conf.name() << " failed.";
      return 1;
    }
  }
  std::thread model_switch_thread(&ModelSelector::Start, model_selector.get());

  // Init metrics
  Metrics metrics;
  metrics.Init();

  // Init predict request queue
  int bench_thread_count = benchmark_config->bench_thread_count();
  if (bench_thread_count <= 0) {
     LOG(ERROR) << "Invalid bench_thread_count = " << bench_thread_count << ", must > 0.";
     return 1;
  }
  std::vector<PredictRequestQueue> predict_request_queue_vec(bench_thread_count);

  // Init producers
  std::vector<std::shared_ptr<PredictRequestProducer>>
      predict_request_producers;
  std::vector<std::thread> producer_threads;
  for (int i = 0; i < bench_thread_count; ++i) {
    std::shared_ptr<PredictRequestProducer> producer =
        std::make_shared<PredictRequestProducer>(&predict_request_queue_vec[i],
                                                 &metrics);
    if (!producer->Init(*benchmark_config, bench_thread_count)) {
      LOG(ERROR) << "PredictRequestProducer init failed.";
      return 1;
    }
    producer_threads.emplace_back(&PredictRequestProducer::Start,
                                  producer.get());
    predict_request_producers.emplace_back(producer);
  }

  // Init consumers
  std::vector<std::shared_ptr<PredictRequestConsumer>>
      predict_request_consumers;
  std::vector<std::thread> consumer_threads;
  int max_queue_size = benchmark_config->max_queue_size();
  if (max_queue_size > 0) max_queue_size = max_queue_size / bench_thread_count + 1;
  for (int i = 0; i < bench_thread_count; ++i) {
    std::shared_ptr<PredictRequestConsumer> consumer =
        std::make_shared<PredictRequestConsumer>(
            model_selector.get(), &predict_request_queue_vec[i], &metrics, max_queue_size);
    consumer_threads.emplace_back(&PredictRequestConsumer::Start, consumer.get());
    predict_request_consumers.emplace_back(consumer);
  }

  metrics.Start();
  int duration = benchmark_config->duration();
  std::this_thread::sleep_for(std::chrono::seconds(duration));
  metrics.Stop();
  model_selector->Stop();

  for (auto& t : consumer_threads) {
    t.join();
  }
  for (auto& t : producer_threads) {
    t.join();
  }
  model_switch_thread.join();

  return 0;
}

bool FileExists(const std::string& conf_dir, std::string* path) {
  if (access(path->c_str(), F_OK) != 0) {
    std::string new_path = conf_dir + *path;
    if (access(new_path.c_str(), F_OK) == 0) {
      *path = new_path;
    } else {
      return false;
    }
  }
  return true;
}
