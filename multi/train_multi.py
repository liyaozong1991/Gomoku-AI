#!/search/odin/huidu/bin/python3
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
from multiprocessing import Manager, Pool
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# log 配置

logging.basicConfig(filename="./logs", level=logging.INFO, format="[%(levelname)s]\t%(asctime)s\tLINENO:%(lineno)d\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

class TrainPipeline():

    def __init__(self, init_model=None):
        # params of the board and the game
        # self.board_width = 10
        # self.board_height = 10
        # self.n_in_row = 5
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.batch_size = 512  # mini-batch size for training
        self.buffer_num = self.batch_size * 100
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 1500
        self.game_batch_num = 1000000000
        self.local_game_batch_num = 20
        self.process_num = 15
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.main_process_wait_time = 300

    def init_tensorflow_net(self, model_file=None):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        logging.info('init tf net')
        from policy_value_net_tensorflow import PolicyValueNet
        policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=model_file)
        policy_value_net.save_model('./current_policy.model')
        logging.info('init tf net finished')

    def collect_selfplay_data_for_multi_threads(self, thread_id, shared_queue, net_lock, data_lock):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(thread_id % 5 + 2)
        logging.info('start selfplay process {}'.format(thread_id))
        def local_thread_func(thread_id, shared_queue, net_lock, data_lock):
            from policy_value_net_tensorflow import PolicyValueNet
            # 读取模型文件，加锁
            logging.info("process {} start {}th selfplay".format(thread_id, index))
            with net_lock:
                logging.info('process {} get net lock'.format(thread_id))
                current_policy = PolicyValueNet(self.board_width, self.board_height, model_file = './current_policy.model')
            logging.info('process {} release net lock'.format(thread_id))
            local_board = Board(width=self.board_width,
                               height=self.board_height,
                               n_in_row=self.n_in_row)
            local_game = Game(local_board)
            local_mcts_player = MCTSPlayer(current_policy.policy_value_fn,
                                               c_puct=self.c_puct,
                                               n_playout=self.n_playout,
                                               is_selfplay=1)
            #for local_id in range(self.local_game_batch_num):
            winner, play_data = local_game.start_self_play(local_mcts_player,
                                                           temp=self.temp)
            play_data = list(play_data)
            play_data = self.get_equi_data(play_data)
            # 添加对弈数据，加锁
            with data_lock:
                logging.info('process {}: get date lock'.format(thread_id))
                shared_queue.extend(play_data)
            logging.info('process {}: release data lock'.format(thread_id))
            logging.info("process {} {}th selfplay finished".format(thread_id, index))
        for index in range(self.game_batch_num):
            pro = multiprocessing.Process(target=local_thread_func, args=(thread_id, shared_queue, net_lock, data_lock))
            pro.start()
            pro.join()
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

    def policy_update(self, current_policy_value_net, shared_queue, net_lock, data_lock):
        """update the policy-value net"""
        with data_lock:
            random_index = list(range(len(shared_queue)))
            random.shuffle(random_index)
            mini_batch = []
            for i in range(self.batch_size):
                mini_batch.append(shared_queue[random_index[i]])
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
        logging.info('update process save model')
        with net_lock:
            logging.info('update process get net lock')
            current_policy_value_net.save_model('./current_policy.model')
        logging.info('update process release net lock')
        logging.info('update process save model finished')
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

    def policy_evaluate(self, best_win_ratio, pure_mcts_playout_num, current_policy_value_net, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
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
        logging.info('update process with pure mcts game start')
        for i in range(n_games):
            logging.info('update process with pure mcts game {} start'.format(i))
            winner = game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
            logging.info('update process with pure mcts game {} finished'.format(i))
        logging.info('update process with pure mcts game all finished'.format(i))
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        logging.info("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        if win_ratio >= best_win_ratio:
            logging.info("New best policy!!!!!!!!")
            best_win_ratio = win_ratio
            # update the best_policy
            current_policy_value_net.save_model('./best_policy.model')
            if (best_win_ratio == 1.0 and
                    pure_mcts_playout_num < 5000):
                pure_mcts_playout_num += 1000
                best_win_ratio = 0.0
        return best_win_ratio, pure_mcts_playout_num

    def update_net(self, shared_queue, net_lock, data_lock, stop_update_process):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        from policy_value_net_tensorflow import PolicyValueNet
        # 读取和写入模型文
        with net_lock:
            current_policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file = './current_policy.model')
        logging.info('update process release net lock')
        i = 0
        best_win_ratio = 0
        pure_mcts_playout_num = 1000
        while stop_update_process.value == 0:
            #with net_lock:
            with data_lock:
                logging.info('update process get data lock')
                shared_queue_length = len(shared_queue)
                while len(shared_queue) > self.buffer_num:
                    shared_queue.pop(0)
            logging.info('update process release data lock')
            if shared_queue_length > self.batch_size:
                logging.info('update process start {} th self train'.format(i))
                loss, entropy = self.policy_update(current_policy_value_net, shared_queue, net_lock, data_lock)
                logging.info('update process end {} th self train'.format(i))
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    logging.info("update process current self-play batch: {}".format(i+1))
                    best_win_ratio, pure_mcts_playout_num = self.policy_evaluate(best_win_ratio, pure_mcts_playout_num, current_policy_value_net)
                i += 1
            time.sleep(1)
        logging.info('update process finished')

    def run(self):
        """run the training pipeline"""
        try:
            # 必须在一个线程中引入tensorflow，否则会造成其他线程由于错误阻塞。
            # ERROR: could not retrieve CUDA device count: CUDA_ERROR_NOT_INITIALIZED
            #init_process = multiprocessing.Process(target=self.init_tensorflow_net, args=('./current_policy.model',))
            init_process = multiprocessing.Process(target=self.init_tensorflow_net)
            init_process.start()
            init_process.join()
            m = Manager()
            shared_queue = m.list()
            net_lock = m.Lock()
            data_lock = m.Lock()
            stop_update_process = multiprocessing.Value('i', 0)
            pro_list = []
            for i in range(self.process_num):
                pro = multiprocessing.Process(target=self.collect_selfplay_data_for_multi_threads, args=(i, shared_queue, net_lock, data_lock))
                pro_list.append(pro)
                pro.start()
            update_process = multiprocessing.Process(target=self.update_net, args=(shared_queue, net_lock, data_lock, stop_update_process))
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
                time.sleep(60)
        except Exception as e:
            logging.error('\n\rquit')

if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    logging.info('start training')
    training_pipeline.run()
    logging.info('all finished')
