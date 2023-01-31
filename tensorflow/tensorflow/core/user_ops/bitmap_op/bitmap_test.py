import os
import tensorflow as tf
import numpy as np
import time
import math
from tensorflow.python.client import timeline

# 0
# 1    2  3
# 4 5  6  7 8 9
set_module =  tf.load_op_library('./bitmap_op.so')
sess = tf.Session()
run_metadata = tf.RunMetadata()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

## init
bitmap = set_module.bitmap_init(np.array([1,4,5,6,7], np.int32), math.ceil(2000000/32))
print("====", sess.run(bitmap))

a = tf.constant(np.random.randint(0, 2000000, 50000, dtype=np.int64), tf.int64)

a_new, bitmap_new = set_module.bitmap_difference(a, bitmap)

res =  sess.run([bitmap, a, a_new, bitmap_new], options=run_options, run_metadata=run_metadata)

print('bitmap:', res[0])
print('a:', res[1])
print('a_new:', res[2])
print('bitmap_new:', res[3])

tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
print(ctf, file=open("timeline","w"))
