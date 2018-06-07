#!/search/odin/liyaozong/tools/python3/bin/python3
# coding: utf8

import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
import multiprocessing
from multiprocessing import Manager, Pool
import time
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_dir='multi_2'
log_name = model_dir + '/logs'
current_model_name = model_dir + '/current_policy_model'
temp_current_model_name = model_dir + '/temp_current_policy_model'
best_model_name = model_dir + '/best_policy_model'

# log 配置
logging.basicConfig(filename=log_name, level=logging.INFO, format="[%(levelname)s]\t%(asctime)s\tLINENO:%(lineno)d\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

class TrainPipeline():

    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        # training params
        self.learn_rate = 2e-3
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.batch_size = 512  # mini-batch size for training
        self.buffer_num = self.batch_size * 100
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 1000
        self.update_freq = 300
        self.game_batch_num = 1000000000
        self.process_num = 12
        self.summary_record_freq = 5
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.main_process_wait_time = 300

    def collect_selfplay_data_thread(self, thread_id, shared_queue, net_lock, data_lock):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(thread_id % 6 + 2)
        def local_thread_func(thread_id, shared_queue, net_lock, data_lock):
            from policy_value_net_tensorflow import PolicyValueNet
            # 读取模型文件，加锁
            logging.info("selfplay process {} ask net lock".format(thread_id))
            with net_lock:
                logging.info('selfpaly process {} get net lock'.format(thread_id))
                current_policy = PolicyValueNet(self.board_width, self.board_height, model_file=current_model_name)
            logging.info('selfplay process {} release net lock'.format(thread_id))
            local_board = Board(width=self.board_width,
                               height=self.board_height,
                               n_in_row=self.n_in_row)
            local_game = Game(local_board)
            local_mcts_player = MCTSPlayer(current_policy.policy_value_fn,
                                          c_puct=self.c_puct,
                                          n_playout=self.n_playout,
                                          is_selfplay=1)
            logging.info("selfplay process {} start {}th selfplay".format(thread_id, index))
            winner, play_data = local_game.start_self_play(local_mcts_player,
                                                           temp=self.temp)
            logging.info("selfplay process {} finish {}th selfplay".format(thread_id, index))
            play_data = list(play_data)
            play_data = self.get_equi_data(play_data)
            # 添加对弈数据，加锁
            logging.info('selfplay process {} ask date lock'.format(thread_id))
            with data_lock:
                logging.info('selfplay process {} get date lock'.format(thread_id))
                shared_queue.extend(play_data)
                while len(shared_queue) > self.buffer_num:
                    shared_queue.pop(0)
            logging.info('selfplay process {} release data lock'.format(thread_id))
        logging.info('selfplay process {} all selfpaly start'.format(thread_id))
        for index in range(self.game_batch_num):
            pro = multiprocessing.Process(target=local_thread_func, args=(thread_id, shared_queue, net_lock, data_lock))
            pro.start()
            pro.join()
        logging.info('selfplay process {} all selfpaly finished'.format(thread_id))

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

    def policy_update(self, current_policy_value_net, shared_queue, net_lock, data_lock, index, lr_multiplier):
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
                    self.learn_rate * lr_multiplier)
            new_probs, new_v = current_policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and lr_multiplier > 0.1:
            lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and lr_multiplier < 10:
            lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        logging.info("update process kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
            kl,
            lr_multiplier,
            loss,
            entropy,
            explained_var_old,
            explained_var_new))
        # summary for tensorboard
        if index % self.summary_record_freq == 0:
            current_policy_value_net.summary_record(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    index,
                    )
        return lr_multiplier

    def update_net_thread(self, shared_queue, net_lock, data_lock, stop_update_process, update_best_model):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        from policy_value_net_tensorflow import PolicyValueNet
        logging.info('update process start')
        # 读取和写入模型文
        current_policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_dir)
        current_policy_value_net.save_model(current_model_name)
        current_policy_value_net.save_model(best_model_name)
        best_win_ratio = 0
        get_enough_train_data = False
        global_update_step = 0
        lr_multiplier = 1.0
        while stop_update_process.value == 0:
            time.sleep(1)
            if get_enough_train_data:
                global_update_step += 1
                logging.info('update process start {} th self train'.format(global_update_step))
                lr_multiplier = self.policy_update(current_policy_value_net, shared_queue, net_lock, data_lock, global_update_step, lr_multiplier)
                logging.info('update process end {} th self train'.format(global_update_step))
                # 这里更新最新模型文件
                logging.info('update process ask net lock')
                with net_lock:
                    logging.info('update process get net lock')
                    current_policy_value_net.save_model(current_model_name)
                logging.info('update process release net lock')
                if (global_update_step + 1) % self.update_freq == 0:
                    update_best_model.value = 1
            else:
                with data_lock:
                    get_enough_train_data = len(shared_queue) >= self.batch_size
        logging.info('update process finished')

    def update_best_model_thread(self, current_model_name, best_model_name, net_lock, update_best_model, stop_update_process):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        def update_best_model_local_process_func(name1, name2):
            from policy_value_net_tensorflow import PolicyValueNet
            PolicyValueNet(self.board_width, self.board_height, model_file=name1).save_model(name2)
        update_best_model_global_num = 0
        update_best_model_time = 0
        logging.info('update best model process start!')
        while stop_update_process.value == 0:
            if update_best_model.value == 1:
                update_best_model_global_num += 1
                logging.info('update best model process global_num:{}'.format(update_best_model_global_num))
                win_num = 0
                start_player = 0
                with net_lock:
                    p = multiprocessing.Process(target=update_best_model_local_process_func, args=(current_model_name, temp_current_model_name))
                    p.start()
                    p.join()
                for i in range(20):
                    board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
                    game = Game(board)
                    logging.info('update best model process start {} th local game, first move player {}'.format(i, start_player))
                    win_player = game.two_net_play(temp_current_model_name, best_model_name, net_lock, start_player=start_player, is_shown=1)
                    logging.info('update best model process {} th local game finished, win player is {}'.format(i, win_player))
                    start_player = 1 - start_player  # play first in turn
                    if win_player == 0:
                        win_num += 1
                logging.info('update best model process global_num:{} finished, current model win total {}'.format(update_best_model_global_num, win_num))
                if win_num >= 11:
                    update_best_model_time += 1
                    logging.info('update best model process get new best model:{}'.format(update_best_model_time))
                    p = multiprocessing.Process(target=update_best_model_local_process_func, args=(temp_current_model_name, best_model_name))
                    p.start()
                    p.join()
                update_best_model.value = 0

    def run(self):
        """run the training pipeline"""
        try:
            # 必须在一个线程中引入tensorflow，否则会造成其他线程由于错误阻塞。
            # ERROR: could not retrieve CUDA device count: CUDA_ERROR_NOT_INITIALIZED
            m = Manager()
            shared_queue = m.list()
            net_lock = m.Lock()
            data_lock = m.Lock()
            stop_update_process = multiprocessing.Value('i', 0)
            update_best_model = multiprocessing.Value('i', 0)
            update_process = multiprocessing.Process(target=self.update_net_thread,
                    args=(shared_queue, net_lock, data_lock, stop_update_process, update_best_model))
            update_process.start()
            time.sleep(5)
            pro_list = []
            for i in range(self.process_num):
                pro = multiprocessing.Process(target=self.collect_selfplay_data_thread, args=(i, shared_queue, net_lock, data_lock))
                pro_list.append(pro)
                pro.start()
                time.sleep(1)
            update_best_model_process = multiprocessing.Process(target=self.update_best_model_thread,
                    args=(current_model_name, best_model_name, net_lock, update_best_model, stop_update_process))
            update_best_model_process.start()
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
                time.sleep(300)
        except Exception as e:
            logging.error('quit')

if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    logging.info('start training')
    training_pipeline.run()
    logging.info('all finished')
