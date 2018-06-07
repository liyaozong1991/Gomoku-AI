# coding: utf8
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
import multiprocessing
import time
from multiprocessing import Process, Manager

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 12))
        self.height = int(kwargs.get('height', 12))
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        # self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be less than {}'.format(self.n_in_row))
        self.start_player = start_player
        self.current_player = start_player  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4 * width * height
        """
        # square_state = np.zeros((4, self.width, self.height))
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            # 己方所有落子位置
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            # 对方所有落子位置
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            # 最后一个子的落子位置，作者说有利于优化训练结果，待验证
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        square_state[3] = self.current_player
        #if len(self.states) % 2 == 0:
        #    square_state[3][:, :] = 1.0  # indicate the colour to play
        #   square_state[2] = self.start_player  # indicate the current player
        #else:
        #    square_state[2] = 1 - self.start_player
        #return square_state[:, ::-1, :]
        return square_state

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = 1 - self.current_player
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row
        if len(states) < n * 2 - 1:
            return False, -1
        player = states[self.last_move]
        self.last_move_height = self.last_move // width
        self.last_move_width =  self.last_move % width

        # '——' 方向判断 
        num = 1
        temp_height = self.last_move_height
        temp_width = self.last_move_width + 1
        while temp_width < width and\
                states.get(self.location_to_move([temp_height, temp_width]), -1) == player:
            num += 1
            temp_width += 1
        temp_height = self.last_move_height
        temp_width = self.last_move_width - 1
        while temp_width >= 0 and\
                states.get(self.location_to_move([temp_height, temp_width]), -1) == player:
            num += 1
            temp_width -= 1
        if num >= self.n_in_row:
            return True, player
        # '|' 方向判断
        num = 1
        temp_height = self.last_move_height + 1
        temp_width = self.last_move_width
        while temp_height < height and\
                states.get(self.location_to_move([temp_height, temp_width]), -1) == player:
            num += 1
            temp_height += 1
        temp_height = self.last_move_height - 1
        temp_width = self.last_move_width
        while temp_height >= 0 and\
                states.get(self.location_to_move([temp_height, temp_width]), -1) == player:
            num += 1
            temp_height -= 1
        if num >= self.n_in_row:
            return True, player
        # '\' 方向判断
        num = 1
        temp_height = self.last_move_height + 1
        temp_width = self.last_move_width + 1
        while temp_width < width and\
                temp_height < height and\
                states.get(self.location_to_move([temp_height, temp_width]), -1) == player:
            num += 1
            temp_width += 1
            temp_height += 1
        temp_height = self.last_move_height - 1
        temp_width = self.last_move_width - 1
        while temp_width >= 0 and\
                temp_height >= 0 and\
                states.get(self.location_to_move([temp_height, temp_width]), -1) == player:
            num += 1
            temp_width -= 1
            temp_height -= 1
        if num >= self.n_in_row:
            return True, player
        # '/' 方向判断
        num = 1
        temp_height = self.last_move_height - 1
        temp_width = self.last_move_width + 1
        while temp_width < width and\
                temp_height >= 0 and\
                states.get(self.location_to_move([temp_height, temp_width]), -1) == player:
            num += 1
            temp_width += 1
            temp_height -= 1
        temp_height = self.last_move_height + 1
        temp_width = self.last_move_width - 1
        while temp_width >= 0 and\
                temp_height < height and\
                states.get(self.location_to_move([temp_height, temp_width]), -1) == player:
            num += 1
            temp_width -= 1
            temp_height += 1
        if num >= self.n_in_row:
            return True, player
        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif len(self.availables) == 0:
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", 0, "with O".rjust(3))
        print("Player", 1, "with X".rjust(3))
        print()
        print(' '*6, end='')
        for x in range(width):
            print("{0:7}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:6d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == 0:
                    print(' '*6, end='')
                    print('\033[0;31;40mO\033[0m', end='')
                elif p == 1:
                    print(' '*6, end='')
                    print('\033[0;36;40mX\033[0m', end='')
                else:
                    print(' '*6, end='')
                    print('_', end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        player1.set_player_ind(0)
        player2.set_player_ind(1)
        players_list = [player1, player2]
        if is_shown:
            self.graphic(self.board)
        while True:
            current_player = players_list[self.board.get_current_player()]
            move = current_player.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players_list[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def two_net_play(self, player1, player2, net_lock, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        def get_net_player_next_action(
                player,
                i,
                shared_board_states,
                shared_board_availables,
                shared_board_last_move,
                shared_board_current_player,
                game_continue,
                winner,
                play_lock,
                net_lock,
                ):
            from policy_value_net_tensorflow import PolicyValueNet
            from mcts_alphaZero import MCTSPlayer
            local_board = Board(width=self.board.width, height=self.board.height, n_in_row=self.board.n_in_row)
            local_board.init_board(start_player)
            with net_lock:
                policy = PolicyValueNet(local_board.width, local_board.height, model_file=player)
            mcts_player = MCTSPlayer(policy.policy_value_fn, c_puct=5, n_playout=400, is_selfplay=0)
            while game_continue.value == 1:
                if shared_board_current_player.value == i:
                    with play_lock:
                        # 必须进行同步，好麻烦
                        for k,v in shared_board_states.items():
                            local_board.states[k] = v
                        local_board.availables = []
                        for availables in shared_board_availables:
                            local_board.availables.append(availables)
                        local_board.last_move = shared_board_last_move.value
                        local_board.current_player = shared_board_current_player.value
                        # 同步结束
                        move = mcts_player.get_action(local_board)
                        local_board.do_move(move)
                        #print('player {} do move {}'.format(i, move))
                        if is_shown:
                            self.graphic(local_board)
                        end, win = local_board.game_end()
                        if end:
                            if win != -1:
                                print("Game end. Winner is", win)
                            else:
                                print("Game end. Tie")
                            game_continue.value = 0
                            winner.value = win
                        # 继续同步
                        shared_board_states[move] = shared_board_current_player.value
                        shared_board_availables.remove(move)
                        shared_board_last_move.value = move
                        shared_board_current_player.value = 1 - shared_board_current_player.value
                time.sleep(0.2)
        game_continue = multiprocessing.Value('i', 1)
        winner = multiprocessing.Value('i', -1)
        m = Manager()
        # play lock
        play_lock = m.Lock()
        # shared board states
        shared_board_states = m.dict()
        shared_board_availables = m.list(range(self.board.width * self.board.height))
        shared_board_last_move = multiprocessing.Value('i', -1)
        shared_board_current_player = multiprocessing.Value('i', start_player)
        best_player_thread = multiprocessing.Process(
                target=get_net_player_next_action,
                args=(
                    player1,
                    0,
                    shared_board_states,
                    shared_board_availables,
                    shared_board_last_move,
                    shared_board_current_player,
                    game_continue,
                    winner,
                    play_lock,
                    net_lock,
                    ),
                )
        current_player_thread = multiprocessing.Process(
                target=get_net_player_next_action,
                args=(
                    player2,
                    1,
                    shared_board_states,
                    shared_board_availables,
                    shared_board_last_move,
                    shared_board_current_player,
                    game_continue,
                    winner,
                    play_lock,
                    net_lock,
                    ),
                )
        best_player_thread.start()
        current_player_thread.start()
        while best_player_thread.is_alive() or current_player_thread.is_alive():
            time.sleep(1)
        return winner.value
            
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
