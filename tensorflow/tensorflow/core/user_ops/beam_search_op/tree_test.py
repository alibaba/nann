import os
import tensorflow as tf
import numpy as np
import time

# 0
# 1    2  3
# 4 5  6  7 8 9
tree = [1, 4, 6, 7, 10]

# 0
# 1      2    3   4      
# 5 6 7  8 9  10  11 12
tree = [1, 5, 8, 10, 11, 13]

# 0         1         2
# 3     4   5         6      7   8
# 9 10  11  12 13 14  15 16  17  18 19 20
tree = [3, 5, 6, 9, 11, 12, 15, 17, 18, 21]
#tree = [-1, -1, -1, 0, 0, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 7, 8, 8, 8]

sess = tf.Session()
with sess.as_default():

  nodes = tf.first_level(tree)
  nodes = nodes.eval()
  print(nodes)
  
  nodes = tf.get_children(nodes, tree)
  nodes = nodes.eval()
  print(nodes)
 
  nodes = tf.get_children(nodes, tree)
  nodes = nodes.eval()
  print(nodes)
 
  nodes = tf.get_parents(nodes, tree)
  nodes = nodes.eval()
  print(nodes)
 
  nodes = tf.get_parents(nodes, tree)
  nodes = nodes.eval()
  print(nodes)

  nodes = tf.get_parents([19, 11, 14, 9, 20, 15, 8, 5], tree) #[8, 4, 5, 3, 8, 6, 2, 1]
  nodes = nodes.eval()
  print(nodes)

