from functools import wraps

import tensorflow as tf

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nann.util import load_meta_graph


def tf_predict(inputs, outputs, meta_path):
  def predict_decorator(func):
    @wraps(func)
    def wrapped_function(*args, **kwargs):
      tf.reset_default_graph()
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.allow_soft_placement = True
      rewrite_options = config.graph_options.rewrite_options
      config.graph_options.rewrite_options.meta_optimizer_timeout_ms = 20000000000
      config.graph_options.rewrite_options.constant_folding = rewrite_options.OFF
      config.graph_options.rewrite_options.arithmetic_optimization = rewrite_options.OFF
      config.graph_options.rewrite_options.layout_optimizer = rewrite_options.OFF
      config.graph_options.rewrite_options.memory_optimization = rewrite_options.NO_MEM_OPT

      with tf.Session(config=config) as sess:
        data = func(*args, **kwargs)
        meta_graph = load_meta_graph(meta_path)
        tf.import_graph_def(meta_graph.graph_def, name="")
        input_tensor_name = [meta_graph.signature_def['predict'].inputs[key].name for key in inputs]
        output_tensor_name = [meta_graph.signature_def['predict'].outputs[key].name for key in outputs]
        output_tensors = [sess.graph.get_tensor_by_name(name) for name in output_tensor_name]
        res = []
        for i in range(len(data)):
          input_dict = {k: v for k, v in zip(input_tensor_name, data[i])}
          res.append(sess.run(output_tensors, input_dict))
        return res

    return wrapped_function

  return predict_decorator
