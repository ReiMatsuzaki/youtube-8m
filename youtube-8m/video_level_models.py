# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags, logging
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
    
    logging.info('MoeModel.create_model begin')
    logging.info('num_mixtures: {0}'.format(num_mixtures))
    
    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}
class MonoModel(models.BaseModel):
  """ Mono layer NN """
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    num_hidden = 1024
    hidden = slim.fully_connected(
      model_input, num_hidden, activation_fn=tf.nn.relu)
    output = slim.fully_connected(
      hidden, vocab_size, activation_fn=tf.nn.softmax)
    return {"predictions": output}

    

class MoNN2LModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-6,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    
    h1Units = 4096
    A1 = slim.fully_connected(
        model_input, h1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H1')
    h2Units = 4096
    A2 = slim.fully_connected(
        A1, h2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H2')
#    
    expert_activations = slim.fully_connected(
        A2,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

class MoNN2LL2Pen8Model(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    
    h1Units = 4096
    A1 = slim.fully_connected(
        model_input, h1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H1')
    h2Units = 4096
    A2 = slim.fully_connected(
        A1, h2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H2')
#    
    expert_activations = slim.fully_connected(
        A2,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

  
class MoNN3LModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-6,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures


    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    
    a1Units = 4096
    A1 = slim.fully_connected(
        model_input, a1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA1')
    a2Units = 4096
    A2 = slim.fully_connected(
        A1, a2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA2')
    a2Units = 4096
    A3 = slim.fully_connected(
        A2, a2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA3')

    expert_activations = slim.fully_connected(
        A3,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}


class CgMoeModel(models.BaseModel):
  """
  CG(Context Gating) is added before the MoE(Mixture of Experts)
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    numx = 128+1024

    w = tf.Variable(tf.truncated_normal([numx,numx], stddev=0.1), name="w")
    b = tf.Variable(tf.zeros([numx]), name="b")
    cg = tf.multiply( tf.nn.sigmoid(tf.matmul(model_input, w) + b),
                      model_input)
                     
    gate_activations = slim.fully_connected(
        cg,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        cg,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}    
    
class old_Cg2MoeModel(models.BaseModel):
  """
  CG(Context Gating) is added before and after the MoE(Mixture of Experts)
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):

    logging.info('Cg2MoeModel.create_model begin')

    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    numx = model_input.get_shape().as_list()[1]
    logging.info('numx: {0}'.format(numx))

    w = tf.Variable(tf.truncated_normal([numx,numx], stddev=0.1), name="w")
    b = tf.Variable(tf.zeros([numx]), name="b")
    cg = tf.multiply( tf.nn.sigmoid(tf.matmul(model_input, w) + b),
                      model_input)
                     
    gate_activations = slim.fully_connected(
        cg,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        cg,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])

    w2 = tf.Variable(tf.truncated_normal([vocab_size,vocab_size], stddev=0.1), name="w2")
    b2 = tf.Variable(tf.zeros([vocab_size]), name="b2")
    cg2 = tf.multiply( tf.nn.sigmoid(tf.matmul(final_probabilities, w2) + b2),
                       final_probabilities)

    return {"predictions": cg2}
    
  
class Cg2MoNN2LModel(models.BaseModel):
  """
  CG(Context Gating) is added before and after MoNN(Mixture of Neural Networks).
  Each NN have two layer.
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    numx = 128+1024

    w = tf.Variable(tf.truncated_normal([numx,numx], stddev=0.1), name="w")
    b = tf.Variable(tf.zeros([numx]), name="b")
    cg = tf.multiply( tf.nn.sigmoid(tf.matmul(model_input, w) + b),
                      model_input)
                     
    gate_activations = slim.fully_connected(
        cg,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")

    a1Units = 4096
    A1 = slim.fully_connected(
      cg,
      a1Units,
      activation_fn=tf.nn.relu,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="FC_HA1")
    a2Units = a1Units
    A2 = slim.fully_connected(
      A1,
      a2Units,
      activation_fn=tf.nn.relu,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="FC_HA2")
    
    expert_activations = slim.fully_connected(
      A2,
      vocab_size * num_mixtures,
      activation_fn=None,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="experts")
    
    gating_distribution = tf.nn.softmax(tf.reshape(
      gate_activations,
      [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])

    w2 = tf.Variable(tf.truncated_normal([vocab_size,vocab_size], stddev=0.1), name="w2")
    b2 = tf.Variable(tf.zeros([vocab_size]), name="b2")
    cg2 = tf.multiply( tf.nn.sigmoid(tf.matmul(final_probabilities, w2) + b2),
                       final_probabilities)

    return {"predictions": cg2}

class Cg2MoNN3LModel(models.BaseModel):
  """
  CG(Context Gating) is added before and after MoNN(Mixture of Neural Networks)
  Each NN have three layer.
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    numx = 128+1024

    w = tf.Variable(tf.truncated_normal([numx,numx], stddev=0.1), name="w")
    b = tf.Variable(tf.zeros([numx]), name="b")
    cg = tf.multiply( tf.nn.sigmoid(tf.matmul(model_input, w) + b),
                      model_input)
                     
    gate_activations = slim.fully_connected(
        cg,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")

    a1Units = 4096
    A1 = slim.fully_connected(
      cg,
      a1Units,
      activation_fn=tf.nn.relu,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="FC_HA1")
    a2Units = a1Units
    A2 = slim.fully_connected(
      A1,
      a2Units,
      activation_fn=tf.nn.relu,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="FC_HA2")
    a3Units = a1Units
    A3 = slim.fully_connected(
      A2,
      a3Units,
      activation_fn=tf.nn.relu,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="FC_HA3")
    
    expert_activations = slim.fully_connected(
      A3,
      vocab_size * num_mixtures,
      activation_fn=None,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="experts")
    
    gating_distribution = tf.nn.softmax(tf.reshape(
      gate_activations,
      [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])

    w2 = tf.Variable(tf.truncated_normal([vocab_size,vocab_size], stddev=0.1), name="w2")
    b2 = tf.Variable(tf.zeros([vocab_size]), name="b2")
    cg2 = tf.multiply( tf.nn.sigmoid(tf.matmul(final_probabilities, w2) + b2),
                       final_probabilities)

    return {"predictions": cg2}
class Cg2MoeModel(models.BaseModel):
  """
  CG(Context Gating) is added before and after the MoE(Mixture of Experts)
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):

    logging.info('Cg2MoeModel.create_model begin')

    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    numx = model_input.get_shape().as_list()[1]
    logging.info('num_mixtures: {0}'.format(num_mixtures))
    logging.info('numx: {0}'.format(numx))

    w1 = tf.get_variable("w1", shape=[numx,numx],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
    b1 = tf.get_variable("b1", shape=[numx],
                        initializer=tf.zeros_initializer())
    cg1 = tf.multiply( tf.nn.sigmoid(tf.matmul(model_input, w1) + b1),
                       model_input)
                     
    gate_activations = slim.fully_connected(
      cg1,
      vocab_size * (num_mixtures + 1),
      activation_fn=None,
      biases_initializer=None,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="gates")
    expert_activations = slim.fully_connected(
        cg1,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])

    w2 = tf.get_variable("w2", shape=[vocab_size,vocab_size],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
    b2 = tf.get_variable("b2", shape=[vocab_size],
                        initializer=tf.zeros_initializer())
    cg2 = tf.multiply( tf.nn.sigmoid(tf.matmul(final_probabilities, w2) + b2),
                       final_probabilities)

    return {"predictions": cg2}
    
