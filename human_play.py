# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#model_file='model_record/best_policy_model_single'
#model_file='model_record/best_policy_model_multi'
#model_file='./single/best_policy_model'
#model_file='./multi/best_policy_model'
#model_file='./multi_2/best_policy_model'
model_file='./multi_2/current_policy_model'

import sys
if len(sys.argv) >= 2:
    model_file = sys.argv[1]


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)

def run():
    n = 5
    width, height = 8, 8
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        best_policy = PolicyValueNet(width, height, model_file = model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=500)

        # human player, input your move in the format: 2,3
        human1 = Human()

        # set start_player=0 for human first
        game.start_play(human1, mcts_player, start_player=1, is_shown=1)
        # game.start_play(human1, human2, start_player=0, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')

if __name__ == '__main__':
    run()
