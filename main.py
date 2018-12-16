#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:17:22 2018

@author: congxin
"""

import argparse
import BiLSTM_CRF

def str2bool(string):
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--embedding_size', type=int, default=300, help='embedding size')
parser.add_argument('--hidden_dim', type=int, default=200, help='number of hidden units')
parser.add_argument('--num_epoch', type=int, default=10000, help='number of epoch')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
args = parser.parse_args()

model = BiLSTM_CRF.BiLSTM_CRF(args)
#model.train()
model.evaluate()