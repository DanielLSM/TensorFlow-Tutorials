'''A complete DDPG agent, everything running on tensorflow should just run 
in this class for sanity and simplicity. Moreoever, every variable and
hyperparameter should be stored within the tensorflow graph to grant
increased performance'''
#agent=DDPG_agent(something,something)....
import tensorflow as tf
import tensorflow.contrib as tc
from models import *
from copy import copy
from math import *
import numpy as np


class DDPG_agent(object):
    
    def __init__(self,observation_dims, action_dims,
        alpha=0.9,gamma=0.99,batch_size=64,tau=5e-4,
        actor_l2_reg=1e-7,critic_l2_reg=1e-7,train_multiplier=1):

        #Pre-processing
        observation_shape = (None,observation_dims)
        action_shape = (None,action_dims)
        print('Inputdims:{}, Outputdims:{}'.format(observation_dims,action_dims))

        #Input tensorflow nodes
        self.observation = tf.placeholder(tf.float32, shape=observation_shape, name='observation')
        self.action = tf.placeholder(tf.float32, shape=action_shape, name='action')        
        self.observation_after = tf.placeholder(tf.float32, shape=observation_shape, name='observation_after')
        self.reward = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')        
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
 
        #Hyper Parameters
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tf.Variable(tau)
        self.actor_l2_reg = actor_l2_reg
        self.critic_l2_reg = critic_l2_reg
        self.batch_size = batch_size

        
        #Networks
        self.actor = Actor(action_dims)
        self.target_actor = copy(self.actor)
        self.target_actor.name = 'target_actor'
        self.critic = Critic(observation_dims,action_dims)
        self.target_critic = copy(self.critic)
        self.target_critic.name = 'target_critic'

        #Expose nodes from the tf graph to be used
        # Critic Nodes
        self.a2 = self.target_actor(self.observation_after)
        self.q2 = self.target_critic(self.observation_after , self.a2)
        self.q1_target = self.reward + (1-self.terminals1) * self.gamma * self.q2
        self.q1_predict = self.critic(self.observation,self.action)
        self.critic_loss = tf.reduce_mean((self.q1_target - self.q1_predict)**2)

        # Actor Nodes
        self.a1_predict = self.actor(self.observation)
        self.q1_predict = self.critic(self.observation,self.a1_predict,reuse=True)
        self.actor_loss = tf.reduce_mean(- self.q1_predict) 

        # Infer
        self.a_infer = self.actor(self.observation,reuse=True)
        self.q_infer = self.critic(self.observation,self.a_infer,reuse=True)

        # Setting Nodes to Sync target networks
        self.setup_target_network_updates()

        # Train Boosters
        self.traincounter = 0

        # Optimzers
        self.opt_actor = tf.train.AdamOptimizer(1e-4)
        self.opt_critic = tf.train.AdamOptimizer(3e-4)

        # L2 weight loss
        #critic_reg_vars = [var for var in self.critic.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
        #self.critic_reg = tc.layers.apply_regularization(
        #    tc.layers.l2_regularizer(self.critic_l2_reg),
        #    weights_list=critic_reg_vars
        #)

        #actor_reg_vars = [var for var in self.actor.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
        #self.actor_reg = tc.layers.apply_regularization(
        #    tc.layers.l2_regularizer(self.actor_l2_reg),
        #    weights_list=actor_reg_vars
        #)

        #Nodes to run one backprop step on the actor and critic
        #self.cstep = self.opt_critic.minimize(self.critic_loss+self.critic_reg,
        #    var_list=self.critic.trainable_vars)
        #self.astep = self.opt_actor.minimize(self.actor_loss+self.actor_reg,
        #    var_list=self.actor.trainable_vars)

        # Nodes to run one backprop step on the actor and critic
        self.cstep = self.opt_critic.minimize(self.critic_loss,var_list=self.critic.trainable_vars)
        self.astep = self.opt_actor.minimize(self.actor_loss,var_list=self.actor.trainable_vars)


        #Saver
        self.saver = tf.train.Saver()
        
        # Initialize and Sync Networks
        self.initialize()
        self.sync_target()

        #A thread lock for all this proxys messing with our agent :)
        import threading as th
        self.lock = th.Lock()

        tf.summary.FileWriter(logdir='DDPG_graph_model', graph=tf.get_default_graph())
        print('agent initialized :>')


    def initialize(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()


    def train(self):
        mem_replay = self.memory_replay
        batch_size = self.batch_size

        if len(self) > batch_size * 128:

            for i in range(self.train_multiplier):
                batch = mem_replay.sample(batch_size)
                sess = tf.get_default_session()
                res = sess.run([self.critic_loss,
                    self.actor_loss,
                    self.cstep,
                    self.astep,
                    self.target_soft_updates],
                    feed_dict={
                    self.observation:batch['obs0'],
                    self.action:batch['actions'],
                    self.observation_after:batch['obs1'],
                    self.reward:batch['rewards'],
                    self.terminals1:batch['terminals_1'],
                    self.tau:5e-4})

        return res

      
   
    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def sync_target(self,update='hard'):
        if update=='hard':
            self.sess.run(self.target_init_updates)
        else:
            self.sess.run(self.target_soft_updates,feed_dict={self.tau:5e-4})


    def fetch_all_tensors(self):
        lista = tf.contrib.graph_editor.get_tensors(tf.get_default_graph())
        print('A lista tem tamanho: ',len(lista))
        return lista


