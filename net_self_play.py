# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_file_single='single/best_policy_model_single'
model_file_multi='multi/best_policy_model_multi'

def run():
    n = 5
    width, height = 8, 8
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # set start_player=0 for single first
        game.two_net_play(model_file_single, model_file_multi, start_player=1)
    except KeyboardInterrupt:
        print('quit')

if __name__ == '__main__':
    run()
