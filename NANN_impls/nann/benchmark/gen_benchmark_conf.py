import argparse
import bench_conf_pb2 
from google.protobuf import text_format
import os

def get_args():
    # Parse commandline
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="TensorFlow variables file to load.")
    return parser.parse_args()

args = get_args()

def main():
    bench_model_config = bench_conf_pb2.BenchModelConfig()
    bench_model_config.name = "nann"
    bench_model_config.frozen_graph = os.path.join(args.model_dir, "exec.pbtxt") 
    bench_model_config.runmeta.append(os.path.join(args.model_dir, "mock.runmeta"))
    bench_model_config.meta_graph = os.path.join(args.model_dir, "exec.meta.pbtxt") ## only used for signature def
    bench_model_config.config_proto = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_proto")
    bench_model_config.predictor_num = 4
    bench_model_config.qps = -1
    bench_model_config.switch_interval = 3000
    bench_model_config.flat_blaze_op = False

    bench_config = bench_conf_pb2.BenchmarkConfig()
    bench_config.bench_model_config.append(bench_model_config)
    bench_config.bench_thread_count = 4
    bench_config.duration = 60

    with open(os.path.join(args.model_dir, "benchmark_conf"), "w") as f:
        f.write(text_format.MessageToString(bench_config))
    print("gen benchconf done!")

if __name__ == "__main__":
    main()
    
