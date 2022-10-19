node {
  name: "blaze_op"
  op: "BlazeXlaOp"
  input: "nick_cate_indicators:0"
  input: "att_ncomm2:0"
  input: "ncomm:0"
  input: "comm:0"
  input: "att_comm:0"
  input: "att_comm1:0"
  input: "att_comm2:0"
  attr {
    key: "InT"
    value {
      list {
        type: DT_INT64
        type: DT_HALF
        type: DT_HALF
        type: DT_HALF
        type: DT_HALF
        type: DT_HALF
        type: DT_HALF
      }
    }
  }
  attr {
    key: "OutT"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
  attr {
    key: "blaze_option_path"
    value {
      s: "/home/jingshan.ljs/tensorflow/tensorflow/core/kernels/blaze_test_data/convert/succ_options"
    }
  }
  attr {
    key: "graph_def"
    value {
      s: "/home/jingshan.ljs/tensorflow/tensorflow/core/kernels/blaze_test_data/convert/ss"
    }
  }
  attr {
    key: "input_names"
    value {
      list {
        s: "nick_cate_indicators"
        s: "att_ncomm2"
        s: "ncomm"
        s: "comm"
        s: "att_comm"
        s: "att_comm1"
        s: "att_comm2"
      }
    }
  }
  attr {
    key: "output_names"
    value {
      list {
        s: "add"
      }
    }
  }
}
node {
  name: "nick_cate_indicators"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
      }
    }
  }
}
node {
  name: "att_ncomm2"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_HALF
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 19200
        }
      }
    }
  }
}
node {
  name: "ncomm"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_HALF
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 2520
        }
      }
    }
  }
}
node {
  name: "comm"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_HALF
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1760
        }
      }
    }
  }
}
node {
  name: "att_comm"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_HALF
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 28800
        }
      }
    }
  }
}
node {
  name: "att_comm1"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_HALF
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 4800
        }
      }
    }
  }
}
node {
  name: "att_comm2"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_HALF
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 10400
        }
      }
    }
  }
}
node {
  name: "add"
  op: "Identity"
  input: "blaze_op"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
versions {
  producer: 134
}

