import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Actor(Model):
    def __init__(self,action_dims,name='actor'):
        super(Actor, self).__init__(name=name)
        self.action_dims = action_dims

    def __call__(self, input_observation, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            
            stddev = 1.
            x = input_observation
            x = tf.layers.dense(x, 256,kernel_initializer=tf.truncated_normal_initializer(mean=0.,stddev=stddev))
            x = lrelu(x)
            
            x = tf.layers.dense(x, 128,kernel_initializer=tf.truncated_normal_initializer(mean=0.,stddev=stddev))
            x = lrelu(x)

            x = tf.layers.dense(x, 128,kernel_initializer=tf.truncated_normal_initializer(mean=0.,stddev=stddev))
            x = lrelu(x)

            x = tf.layers.dense(x, 128,kernel_initializer=tf.truncated_normal_initializer(mean=0.,stddev=stddev))
            x = lrelu(x)
            
            x = tf.layers.dense(x, self.action_dims, kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=stddev))
            x = tf.nn.tanh(x)

        return x

class Critic(Model):
    def __init__(self,observation_dims,action_dims,name='critic'):
        super(Critic, self).__init__(name=name)

    def __call__(self, input_observation, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            
            stddev = 1.
            x = input_observation
            x = tf.layers.dense(x, 256,kernel_initializer=tf.truncated_normal_initializer(mean=0.,stddev=stddev))
            x = lrelu(x)

            x = tf.layers.dense(x, 128,kernel_initializer=tf.truncated_normal_initializer(mean=0.,stddev=stddev))
            x = lrelu(x)  

            x = tf.layers.dense(x, 128,kernel_initializer=tf.truncated_normal_initializer(mean=0.,stddev=stddev))
            x = lrelu(x)          

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 128,kernel_initializer=tf.truncated_normal_initializer(mean=0.,stddev=stddev))
            x = lrelu(x)

            x = tf.layers.dense(x, 64,kernel_initializer=tf.truncated_normal_initializer(mean=0.,stddev=stddev))
            x = lrelu(x)   

            x = tf.layers.dense(x, 1, kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=stddev))
        return x 

def get_target_updates(vars, target_vars, tau):
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)