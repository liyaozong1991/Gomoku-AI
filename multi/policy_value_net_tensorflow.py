# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow
Tested in Tensorflow 1.4 and 1.5
"""

import numpy as np
import tensorflow as tf
import logging

logging.basicConfig(filename="./logs", level=logging.INFO, format="[%(levelname)s]\t%(asctime)s\tLINENO:%(lineno)d\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

class PolicyValueNet():
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        # Make a session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = False
        self.session = tf.Session(config=config)

        if model_file is not None:
            logging.info('import meta graph')
            # For saving and restoring
            self.saver = tf.train.import_meta_graph(model_file + '-10.meta')
            logging.info('restore graph')
            self.saver.restore(self.session, tf.train.latest_checkpoint('./'))
            logging.info('model have loaded')
        else:
            logging.info('init model')
            # Define the tensorflow neural network
            # 1. Input:
            self.input_states = tf.placeholder(tf.float32,
                                               shape=[None, 4, board_height, board_width],
                                               name='input_states')
            self.input_states_reshaped = tf.reshape(self.input_states,
                                                    [-1, board_height, board_width, 4],
                                                    name='input_states_reshaped')
            # 2. Common Networks Layers
            self.conv1 = tf.layers.conv2d(inputs=self.input_states_reshaped,
                                          filters=32,
                                          kernel_size=[3, 3],
                                          padding="same",
                                          activation=tf.nn.relu,
                                          name='conv1')
            self.conv2 = tf.layers.conv2d(inputs=self.conv1,
                                          filters=64,
                                          kernel_size=[3, 3],
                                          padding="same",
                                          activation=tf.nn.relu,
                                          name='conv2')
            self.conv3 = tf.layers.conv2d(inputs=self.conv2,
                                          filters=128,
                                          kernel_size=[3, 3],
                                          padding="same",
                                          activation=tf.nn.relu,
                                          name='conv3')
            # 3-1 Action Networks
            self.action_conv = tf.layers.conv2d(inputs=self.conv3,
                                                filters=4,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                activation=tf.nn.relu,
                                                name='action_conv')
            # Flatten the tensor
            self.action_conv_flat = tf.reshape(self.action_conv,
                                               [-1, 4 * board_height * board_width],
                                               name='action_conv_flat')
            # 3-2 Full connected layer, the output is the log probability of moves
            # on each slot on the board
            self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                             units=board_height * board_width,
                                             activation=tf.nn.log_softmax,
                                             name='action_fc')
            # 4 Evaluation Networks
            self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3,
                                                    filters=2,
                                                    kernel_size=[1, 1],
                                                    padding="same",
                                                    activation=tf.nn.relu,
                                                    name='evaluation_conv')

            self.evaluation_conv_flat = tf.reshape(self.evaluation_conv,
                                                   [-1, 2 * board_height * board_width],
                                                   name='evaluation_conv_flat')

            self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                                  units=64,
                                                  activation=tf.nn.relu,
                                                  name='evaluation_fc1')
            # output the score of evaluation on current state
            self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                                  units=1,
                                                  activation=tf.nn.tanh,
                                                  name='evaluation_fc2')

            # Define the Loss function
            # 1. Label: the array containing if the game wins or not for each state
            self.labels = tf.placeholder(tf.float32,
                                         shape=[None, 1],
                                         name='labels')
            # 2. Predictions: the array containing the evaluation score of each state
            # which is self.evaluation_fc2
            # 3-1. Value Loss function
            self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                           self.evaluation_fc2,
                                                           scope='value_loss')
            # 3-2. Policy Loss function
            self.mcts_probs = tf.placeholder(tf.float32,
                                             shape=[None, board_height * board_width],
                                             name='mcts_probs')

            self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.mcts_probs,
                                                                                    self.action_fc),
                                                                        1)),
                                           name='policy_loss')
            # 3-3. L2 penalty (regularization)
            l2_penalty_beta = 1e-4
            vars = tf.trainable_variables()
            l2_penalty = tf.multiply(l2_penalty_beta,
                                     tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()]))
            # 3-4 Add up to be the Loss function
            self.loss = tf.add(tf.add(self.value_loss,
                                      self.policy_loss),
                               l2_penalty,
                               name='add_loss')

            # Define the optimizer we use for training
            self.learning_rate = tf.placeholder(tf.float32,
                                                name='learning_rate')

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                               name='optimizer')


            # calc policy entropy, for monitoring only
            self.entropy = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)),
                                       name='entropy')

            #for ss in tf.get_default_graph().as_graph_def().node:
            #    print(ss.name)
            # Initialize variables
            init = tf.global_variables_initializer()
            self.session.run(init)

            # For saving and restoring
            self.saver = tf.train.Saver(max_to_keep=2)


    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        #log_act_probs, value = self.session.run(
        #        [self.action_fc, self.evaluation_fc2],
        #        feed_dict={self.input_states: state_batch}
        #        )
        #act_probs = np.exp(log_act_probs)
        log_act_probs, value = self.session.run(
                ["action_fc/LogSoftmax:0", "evaluation_fc2/Tanh:0"],
                feed_dict={"input_states:0": state_batch}
                )
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy = self.session.run(
                ["add_loss:0", "entropy:0"],
                feed_dict={"input_states:0": state_batch,
                           "mcts_probs:0": mcts_probs,
                           "labels:0": winner_batch,
                           "learning_rate:0": lr})
        return loss, entropy

    def save_model(self, model_path, write_meta_graph=True):
        #self.saver.save(self.session, model_path)
        self.saver.save(self.session, model_path, write_meta_graph=write_meta_graph, global_step=10)

    def destroy_model(self):
        self.session.close()
