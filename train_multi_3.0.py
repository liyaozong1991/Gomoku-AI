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
from tensorflow.python.platform import gfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_dir='./multi_3'
log_name = model_dir + '/logs'
current_model_name = model_dir + '/current_policy_model'
temp_current_model_name = model_dir + '/temp_current_policy_model'
best_model_name = model_dir + '/best_policy_model'


class TrainPipeline():

    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        # training params
        self.learn_rate = 2e-3
        # self.lr_multiplier = 1.0
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.batch_size = 512  # mini-batch size for training
        self.buffer_num = self.batch_size * 10
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 60
        self.game_batch_num = 1000000000
        self.process_num = 12
        self.summary_record_freq = 5
        # self.global_update_step = 0

    def collect_selfplay_data_thread(self, thread_id, shared_queue, net_lock, data_lock):
        logging.info('selfpaly process start: {}'.format(thread_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(thread_id % 6 + 2)
        from policy_value_net_tensorflow import PolicyValueNet
        # 读取模型文件，加锁
        with net_lock:
            current_policy = PolicyValueNet(self.board_width, self.board_height, model_file=current_model_name)
        local_board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
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
        with data_lock:
            shared_queue.extend(play_data)
            while len(shared_queue) > self.buffer_num:
                shared_queue.pop(0)
        logging.info('selfpaly process finished: {}'.format(thread_id))

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

    def policy_update(self, current_policy_value_net, shared_queue, net_lock, global_update_step, lr_multiplier):
        """update the policy-value net"""
        random_index = list(range(len(shared_queue)))
        random.shuffle(random_index)
        mini_batch = []
        for i in range(self.batch_size):
            mini_batch.append(shared_queue[random_index[i]])
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = current_policy_value_net.policy_value(state_batch)
        loss, entropy = current_policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * lr_multiplier.value)
        new_probs, new_v = current_policy_value_net.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1)
        )
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and lr_multiplier.value > 0.1:
            lr_multiplier.value /= 1.5
        elif kl < self.kl_targ / 2 and lr_multiplier.value < 10:
            lr_multiplier.value *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        logging.info("update current model process kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
            kl,
            lr_multiplier.value,
            loss,
            entropy,
            explained_var_old,
            explained_var_new))
        # summary for tensorboard
        if global_update_step.value % self.summary_record_freq == 0:
            current_policy_value_net.summary_record(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    global_update_step.value,
                    )

    def update_net(self, shared_queue, net_lock, update_best_model, global_update_step, lr_multiplier, stop_update_process, update_or_selfplay):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        from policy_value_net_tensorflow import PolicyValueNet
        current_policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_dir)
        current_policy_value_net.save_model(current_model_name)
        current_policy_value_net.save_model(best_model_name)
        while global_update_step.value <= self.game_batch_num:
            if update_or_selfplay.value == 0:
                if len(shared_queue) >= self.batch_size:
                    for _ in range(self.epochs):
                        global_update_step.value += 1
                        logging.info('update current model process start self train: {}'.format(global_update_step.value))
                        self.policy_update(current_policy_value_net, shared_queue, net_lock, global_update_step, lr_multiplier)
                        if (global_update_step.value) % self.check_freq == 0:
                            update_best_model.value = 1
                    # 这里更新最新模型文件
                    with net_lock:
                        logging.info('update process update current model')
                        current_policy_value_net.save_model(current_model_name)
                update_or_selfplay.value = 1
            else:
                time.sleep(1)
        stop_update_process.value = 1


    def update_best_model_thread(self, current_model_name, best_model_name, net_lock, update_best_model, stop_update_process):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        def update_best_model_local_process_func(name1, name2):
            from policy_value_net_tensorflow import PolicyValueNet
            PolicyValueNet(self.board_width, self.board_height, model_file=name1).save_model(name2)
        update_best_model_global_num = 0
        update_best_model_time = 0
        logging.info('update best model process start!')
        while stop_update_process.value == 0:
            time.sleep(1)
            if update_best_model.value == 1:
                update_best_model.value = 0
                update_best_model_global_num += 1
                logging.info('update best model process global_num:{}'.format(update_best_model_global_num))
                win_num = 0
                start_player = 0
                with net_lock:
                    p = multiprocessing.Process(target=update_best_model_local_process_func, args=(current_model_name, temp_current_model_name))
                    p.start()
                    p.join()
                for i in range(30):
                    board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
                    game = Game(board)
                    win_player = game.two_net_play(temp_current_model_name, best_model_name, net_lock, start_player=start_player, is_shown=0)
                    start_player = 1 - start_player  # play first in turn
                    if win_player == 0:
                        win_num += 1
                logging.info('update best model process global_num:{} finished, current model win total {}'.format(update_best_model_global_num, win_num))
                if win_num >= 16:
                    update_best_model_time += 1
                    logging.info('update best model process get new best model:{}'.format(update_best_model_time))
                    p = multiprocessing.Process(target=update_best_model_local_process_func, args=(temp_current_model_name, best_model_name))
                    p.start()
                    p.join()

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
            global_update_step = multiprocessing.Value('i', 0)
            lr_multiplier = multiprocessing.Value('d', 1.0)
            update_or_selfplay = multiprocessing.Value('i', 0)
            update_best_model_process = multiprocessing.Process(target=self.update_best_model_thread,
                    args=(current_model_name, best_model_name, net_lock, update_best_model, stop_update_process))
            update_best_model_process.start()
            update_current_model_process = multiprocessing.Process(target=self.update_net,
                    args=(shared_queue, net_lock, update_best_model, global_update_step, lr_multiplier, stop_update_process, update_or_selfplay))
            update_current_model_process.start()
            while stop_update_process.value == 0:
                if update_or_selfplay.value == 1:
                    # 产生训练数据
                    pro_list = []
                    for i in range(self.process_num):
                        pro = multiprocessing.Process(target=self.collect_selfplay_data_thread, args=(i, shared_queue, net_lock, data_lock))
                        pro_list.append(pro)
                        pro.start()
                    for pro in pro_list:
                        pro.join()
                    update_or_selfplay.value = 0
                else:
                    time.sleep(0.1)
            # 等待更新最有模型的进程结束，再结束主程序
            stop_update_process.value = 1
            while update_best_model_process.is_alive():
                time.sleep(5)
        except Exception as e:
            logging.error(e)

if __name__ == '__main__':
    if gfile.Exists(model_dir):
        gfile.DeleteRecursively(model_dir)
    gfile.MakeDirs(model_dir)
    # log 配置
    logging.basicConfig(filename=log_name, level=logging.INFO, format="[%(levelname)s]\t%(asctime)s\tLINENO:%(lineno)d\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    training_pipeline = TrainPipeline()
    logging.info('start training')
    training_pipeline.run()
    logging.info('all finished')

