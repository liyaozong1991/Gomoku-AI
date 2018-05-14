# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku
@author: Junxiao Song
"""

import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
import logging
import multiprocessing
import time
#from multiprocessing.managers import SyncManager
from multiprocessing import Manager, Pool
import importlib
from itertools import repeat

# log 配置
logging.basicConfig(filename="./logs", level=logging.INFO, format="[%(levelname)s]\t%(asctime)s\tLINENO:%(lineno)d\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

class TrainPipeline():

    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 10
        self.board_height = 10
        self.n_in_row = 5
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 4  # num of simulations for each move
        self.c_puct = 5
        self.batch_size = 512  # mini-batch size for training
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 100
        self.process_num = 2
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.main_process_wait_time = 15
        self.net_lock = multiprocessing.Lock()
        self.data_lock = multiprocessing.Lock()

    def init_tensorflow_net(self):
        logging.info('init tf net')
        from policy_value_net_tensorflow import PolicyValueNet
        policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        policy_value_net.save_model('./current_policy.model')
        logging.info('init tf net finished')

    def collect_selfplay_data_for_multi_threads(self, thread_id, shared_queue):
        from policy_value_net_tensorflow import PolicyValueNet
        logging.info('start selfplay process {}'.format(thread_id))
        for index in range(self.game_batch_num):
        # 读取模型文件，加锁
            logging.info("process {} start {}th selfplay".format(thread_id, index))
            with self.net_lock:
                current_policy = PolicyValueNet(self.board_width, self.board_height, model_file = './current_policy.model')
            local_board = Board(width=10,
                               height=10,
                               n_in_row=5)
            local_game = Game(local_board)
            local_mcts_player = MCTSPlayer(current_policy.policy_value_fn,
                                               c_puct=self.c_puct,
                                               n_playout=self.n_playout,
                                               is_selfplay=1)
            winner, play_data = local_game.start_self_play(local_mcts_player,
                                                           temp=self.temp)
            play_data = list(play_data)
            play_data = self.get_equi_data(play_data)
            # 添加对弈数据，加锁
            #with self.data_lock:
            #print('-----------')
            #print(thread_id)
            #print(len(play_data))
            shared_queue.extend(play_data)
            #shared_queue_length.value = min(self.batch_size * 10, shared_queue_length.value + len(play_data))
            #shared_queue_length.value += len(play_data)
            #print(shared_queue_length.value)
            logging.info("process {} {}th selfplay finished".format(thread_id, index))
        logging.info('process {} all selfpaly finished'.format(thread_id))

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def policy_update(self, shared_queue, net_lock, data_lock, pure_mcts_playout_num):
        from policy_value_net_tensorflow import PolicyValueNet
        # 读取和写入模型文件，加锁
        current_policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file = './current_policy.model')
        """update the policy-value net"""
        #random_index = list(range(shared_queue_length.value))
        #random.shuffle(random_index)
        #mini_batch = []
        #for i in range(self.batch_size):
        #    mini_batch.append(shared_queue[random_index[i]])
        mini_batch = random.sample(shared_queue, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = current_policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = current_policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = current_policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
            # 这里更新了模型文件
            current_policy_value_net.save_model('./current_policy.model')
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        logging.info(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, pure_mcts_playout_num, n_games=10):
        from policy_value_net_tensorflow import PolicyValueNet
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file = './current_policy.model')
        current_mcts_player = MCTSPlayer(current_policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        board = Board(width=self.board_width,
                      height=self.board_height,
                      n_in_row=self.n_in_row)
        game = Game(board)
        for i in range(n_games):
            winner = game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        logging.info("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def update_net(self, shared_queue, net_lock, data_lock, stop_update_process):
        from policy_value_net_tensorflow import PolicyValueNet
        i = 0
        best_win_ratio = 0
        pure_mcts_playout_num = 1000
        while stop_update_process.value == 0:
            with net_lock:
                #with data_lock:
                if len(shared_queue) > self.batch_size:
                    loss, entropy = self.policy_update(shared_queue, net_lock, data_lock, pure_mcts_playout_num)
                    # check the performance of the current model,
                    # and save the model params
                    i += 1
                    if (i+1) % self.check_freq == 0:
                        logging.info("current self-play batch: {}".format(i+1))
                        win_ratio = self.policy_evaluate()
                        if win_ratio > best_win_ratio:
                            logging.info("New best policy!!!!!!!!")
                            best_win_ratio = win_ratio
                            # update the best_policy
                            PolicyValueNet(self.board_width, self.board_height, model_file = './current_policy.model').save_model('./best_policy.model')
                            if (best_win_ratio == 1.0 and
                                    pure_mcts_playout_num < 5000):
                                pure_mcts_playout_num += 1000
                                best_win_ratio = 0.0
            time.sleep(4)
        logging.info('update net process finished')

    def run(self):
        """run the training pipeline"""
        try:
            # 必须在一个线程中引入tensorflow，否则会造成其他线程由于错误阻塞。
            # ERROR: could not retrieve CUDA device count: CUDA_ERROR_NOT_INITIALIZED
            init_process = multiprocessing.Process(target=self.init_tensorflow_net)
            init_process.start()
            init_process.join()
            #shared_queue = m.deque(maxlen=self.batch_size * 10)
            shared_queue = Manager().list()
            #shared_queue_length= multiprocessing.Value('i', 0)
            stop_update_process = multiprocessing.Value('i', 0)
            pro_list = []
            for i in range(self.process_num):
                pro = multiprocessing.Process(target=self.collect_selfplay_data_for_multi_threads, args=(i, shared_queue))
                pro_list.append(pro)
                pro.start()
            update_process = multiprocessing.Process(target=self.update_net, args=(shared_queue,self.net_lock, self.data_lock, stop_update_process))
            update_process.start()
            # 保证模型基本启动完成
            time.sleep(self.main_process_wait_time)
            all_finished = True
            while update_process.is_alive():
                for pro in pro_list:
                    if pro.is_alive():
                        all_finished = False
                        break
                if all_finished:
                    stop_update_process.value = 1
                #children_list = multiprocessing.active_children()
                #print("children num:{}".format(len(children_list)))
                #if len(children_list) >= 2: # 正常运行
                #    time.sleep(1)
                #elif len(children_list) == 1: # 对弈进程全部结束，准备更新网络的进程
                #    stop_update_process.value = 1
                #    time.sleep(1)
                #else: # 全部进程结束，准确退出
                #    break
        except KeyboardInterrupt:
            logging.error('\n\rquit')

if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    logging.info('start training')
    training_pipeline.run()
    logging.info('all finished')
