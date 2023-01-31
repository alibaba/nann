import numpy as np
import tensorflow as tf

from nann.logger import get_logger

logger = get_logger(__name__)


def prelu(x, scope='prelu'):
  _alpha = tf.get_variable(scope, shape=[x.get_shape()[-1]], dtype=x.dtype, initializer=tf.constant_initializer(0.25))
  return tf.maximum(0.0, x) + _alpha * tf.minimum(0.0, x)


class LayerNorm:
  def __init__(self, epsilon, scope="ln"):
    self.epsilon = epsilon
    self.scope = scope

  def call(self, inputs):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      inputs_shape = inputs.get_shape()
      params_shape = inputs_shape[-1:]
      mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
      beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
      gamma = tf.get_variable("gama", params_shape, initializer=tf.ones_initializer())
      normalized = (inputs - mean) / ((variance + self.epsilon) ** 0.5)
      outputs = gamma * normalized + beta

    return outputs


class DNN(object):
  def __init__(self, output_dim, use_bias=True, active_op=None, norm_op=None, name='default'):
    self.use_bias = use_bias
    self.active_op = active_op
    self.name = name
    self.output_dim = output_dim
    self.norm_op = norm_op
    if self.norm_op == "ln":
      self.ln = LayerNorm(epsilon=1e-6)

  def call(self, bottom_data, training=True, fc_name='fc'):
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      out = tf.layers.dense(
        bottom_data, self.output_dim,
        use_bias=self.use_bias,
        activation=None, name=fc_name, reuse=tf.AUTO_REUSE,
        kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in', distribution='normal'),
        bias_initializer=tf.constant_initializer(0.1))

      if self.norm_op is not None:
        if self.norm_op == 'bn':
          out = tf.layers.batch_normalization(out, name='bn', reuse=tf.AUTO_REUSE, training=training)
        elif self.norm_op == 'ln':
          out = self.ln.call(out)
        else:
          raise NotImplementedError("normalization %s not supported" % self.norm_op)

      if self.active_op is not None:
        if self.active_op == 'prelu':
          out = prelu(out, scope='prelu')
        elif self.active_op == 'relu':
          out = tf.nn.relu(out)
        else:
          raise NotImplementedError("activation %s not supported" % self.active_op)

      return out


def nonlinear_attention(q, k, v):
  """
  q: shape (#k, n, query_dim=128)
  k: shape (#k, seq_len, emb_dim=128)
  v: same shape with k
  return:
    shape (#k, n, seq_len, emb_dim)
  """
  with tf.variable_scope('nonlinear_attention', reuse=tf.AUTO_REUSE):
    emb_dim = k.get_shape().as_list()[-1]

    q = prelu(tf.layers.dense(q, emb_dim * 2, use_bias=True), 'prelu_q')
    q_ = tf.layers.dense(q, emb_dim * 4, use_bias=True)

    k = prelu(tf.layers.dense(k, emb_dim * 2, use_bias=True), 'prelu_k')
    k_ = tf.layers.dense(k, emb_dim * 4, use_bias=True)

    with tf.variable_scope('scale_dot_product', reuse=tf.AUTO_REUSE):
      d_k = q_.get_shape().as_list()[-1]

      att = tf.einsum('knd,kld->knl', q_, k_)
      att /= d_k ** 0.5

      softmax_att = tf.nn.softmax(att, axis=-1)

      att_out = softmax_att[:, :, :, tf.newaxis] * v[:, tf.newaxis, :, :]

  return att_out


def kl_divergence_with_logit(q_logit, p_logit):
  q = tf.nn.softmax(q_logit, axis=-1)
  qlogq = tf.reduce_mean(tf.reduce_sum(q * tf.nn.log_softmax(q_logit, axis=-1), axis=-1))
  qlogp = tf.reduce_mean(tf.reduce_sum(q * tf.nn.log_softmax(p_logit, axis=-1), axis=-1))
  return qlogq - qlogp


def huge_constant(path, dtype=None):
  with open(path, 'rb') as f:
    _ = np.lib.format.read_magic(f)
    shape, fortran, original_dtype = np.lib.format.read_array_header_1_0(f)

  if dtype is None:
    dtype = original_dtype
  else:
    np_dtype = dtype.as_numpy_dtype
    if np_dtype != original_dtype:
      logger.info(f"cast {path} from {original_dtype} to {np_dtype}")
      arr = np.load(path).astype(np_dtype)
      np.save(path, arr)

  return tf.huge_const(path=path, dtype=dtype, shape=shape)


def tf_print(prefix, var):
  print_op = tf.print(prefix, var)
  with tf.control_dependencies([print_op]):
    return tf.identity(var)


def accuracy(logits, label):
  correct = tf.equal(tf.argmax(logits, axis=-1), tf.argmax(label, axis=-1))
  return tf.reduce_mean(tf.cast(correct, tf.float32))
