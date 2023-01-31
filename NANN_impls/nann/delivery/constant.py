# config for native delivery

native_delivery_input_prefix = 'inference_feed_inputs/'
native_delivery_output_prefix = 'inference_fetch_outputs/'

native_delivery_config = {
  "target_graph_def": "frozen_graph.pb",
  "target_opt_conf_path": "opt_default.conf",
  'inputs': [native_delivery_input_prefix + "user_seq_emb",
             native_delivery_input_prefix + "item_emb"],
  'outputs': [native_delivery_output_prefix + "logits"]
}
