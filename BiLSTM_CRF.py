#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:22:00 2018

@author: congxin
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import re
import time
from sklearn.metrics import precision_recall_fscore_support
import operator
from functools import reduce

import DataLoader

class BiLSTM_CRF(object):
    def __init__(self, args):
        self.loader = DataLoader.DataLoader()
        self.vocab_size = self.loader.vocab_size
        self.label_size = self.loader.label_size
        
        self.embedding_size = args.embedding_size
        self.hidden_dim = args.hidden_dim
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.learning_rate =args.learning_rate
        
        self.save_prediction_file = "./data/prediction.txt"

    def createGraph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # input data
            self.train_data = tf.placeholder(tf.int32, shape=[None, None], name="train_data")
            self.train_labels = tf.placeholder(tf.int32, shape=[None, None], name="train_lables")
            self.train_lengths = tf.placeholder(tf.int32, shape=[None], name="train_lengths")
            
            # embedding
            embedding_table = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
            embeddings = tf.nn.embedding_lookup(params=embedding_table, ids=self.train_data)
            
            # bi-lstm
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            (fw_seq, bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, embeddings, sequence_length=self.train_lengths, dtype=tf.float32
                    )
            output = tf.concat([fw_seq, bw_seq], axis=-1)
            
            # score
            W = tf.get_variable(name="W", 
                                shape=[2*self.hidden_dim, self.label_size], 
                                initializer=tf.contrib.layers.xavier_initializer(), 
                                dtype=tf.float32)
            b = tf.get_variable(name="b", 
                                shape=[self.label_size], 
                                initializer=tf.zeros_initializer(), 
                                dtype=tf.float32)
            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b
        #    unary_scores = tf.reshape(pred, [batch_size, -1, label_size])
            self.unary_scores = tf.reshape(pred, [-1, s[1], self.label_size], name="unary_scores")
            
            # loss
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                    inputs=self.unary_scores, 
                    tag_indices=self.train_labels, 
                    sequence_lengths=self.train_lengths)
            self.loss = tf.reduce_mean(-self.log_likelihood)
            
            # optimizer
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
    def train(self):
        self.createGraph()
        
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.gpu_options.allow_growth = True
        with tf.Session(graph=self.graph) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            print('Initialized')
          
            for step in range(self.num_epoch):
                start = time.time()
                batch_sentences, batch_labels, batch_sentences_lengths = self.loader.getTrainBatch(self.batch_size)
                feed_dict = {self.train_data : batch_sentences, 
                             self.train_labels : batch_labels, 
                             self.train_lengths : batch_sentences_lengths}
                l, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
                print("{}/{} train_loss : {:.3f}, time/batch = {:.3f}".format(step, self.num_epoch, l, time.time()-start))
                
                if (step + 1) % 50 == 0:
                    saver.save(sess, "./model/model")
                    self.test(sess)
            
            saver.save(sess, "./model/model")
                
    def test(self, sess):
        test_batch_sentences, test_batch_labels, test_sentences_lengths = self.loader.getTestBatch(self.batch_size)
#        print(np.isnan(np.array(test_batch_sentences)).any())
        test_feed_dict = {
                self.train_data : test_batch_sentences, 
                self.train_labels : test_batch_labels, 
                self.train_lengths : test_sentences_lengths
                }
        t1 = sess.run([self.unary_scores], test_feed_dict)
        t2 = sess.run([self.transition_params], test_feed_dict)
        y_true = []
        y_pred = []
        for cnt, line in enumerate(t1[0]):
            t4 = line[:test_sentences_lengths[cnt]]
            t3, t5 = tf.contrib.crf.viterbi_decode(t4, t2[0])
            y_true.extend(test_batch_labels[cnt][:test_sentences_lengths[cnt]])
            y_pred.extend(t3[:test_sentences_lengths[cnt]])
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        df = pd.DataFrame(np.array([precision, recall, fscore]).T, columns=["Precision", "Recall", "F1"], index=list(self.loader.id2label))
#        np.set_printoptions(precision=3, suppress=True)
        print(df)
        print()
    
    def evaluate(self):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("model/model.meta")
            saver.restore(sess,tf.train.latest_checkpoint("model/"))
            graph = tf.get_default_graph()
            
            # print all tensors' name
#            tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node] 
#            for tensor_name in tensor_name_list: 
#                print(tensor_name,'\n')
                
            self.train_data = graph.get_tensor_by_name("train_data:0")
            self.train_labels = graph.get_tensor_by_name("train_lables:0")
            self.train_lengths = graph.get_tensor_by_name("train_lengths:0")
            self.unary_scores = graph.get_tensor_by_name("Reshape_1:0")
            self.transition_params = graph.get_tensor_by_name("transitions:0")
            
            y_true = []
            y_pred = []
            save_seq = []
            
            test_batch_sentences, test_batch_labels, test_sentences_lengths = self.loader.getAllTestData()
            for i, j in zip([0, 100, 200, 300], [100, 200, 300, 309]):
                eval_batch_sentences = test_batch_sentences[i:j]
                eval_batch_labels = test_batch_labels[i:j]
                eval_sentences_lengths = test_sentences_lengths[i:j]
                eval_feed_dict = {
                        self.train_data : eval_batch_sentences, 
                        self.train_labels : eval_batch_labels, 
                        self.train_lengths : eval_sentences_lengths
                        }
                t1 = sess.run([self.unary_scores], eval_feed_dict)
                t2 = sess.run([self.transition_params], eval_feed_dict)
                for cnt, line in enumerate(t1[0]):
                    t4 = line[:eval_sentences_lengths[cnt]]
                    t3, t5 = tf.contrib.crf.viterbi_decode(t4, t2[0])
                    y_true.extend(eval_batch_labels[cnt][:eval_sentences_lengths[cnt]])
                    y_pred.extend(t3[:eval_sentences_lengths[cnt]])
                    save_seq.append(t3[:eval_sentences_lengths[cnt]])
#            precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=None)
#            df = pd.DataFrame(np.array([precision, recall, fscore]).T, columns=["Precision", "Recall", "F1"], index=list(self.loader.id2label))
#            print(df)
            
        self.savePrediction(save_seq)
            
#        expr_1 = r"({}{}*|{}{}*|{}{}*)".format(
#                self.loader.label2id["B-ORG"], self.loader.label2id["I-ORG"], 
#                self.loader.label2id["B-PER"], self.loader.label2id["I-PER"], 
#                self.loader.label2id["B-LOC"], self.loader.label2id["I-LOC"])
#        expr_2 = r"{}+".format(self.loader.label2id["O"])

        expr = [r"{}{}*".format(self.loader.label2id["B-ORG"], self.loader.label2id["I-ORG"]),
                r"{}{}*".format(self.loader.label2id["B-PER"], self.loader.label2id["I-PER"]),
                r"{}{}*".format(self.loader.label2id["B-LOC"], self.loader.label2id["I-LOC"])]
        
        y_true_strings = reduce(operator.add, map(str, y_true))
        y_pred_strings = reduce(operator.add, map(str, y_pred))
        precision = [0] * 3
        recall = [0] * 3
        f1 = [0] * 3
        
        b_i_dict = {str(self.loader.label2id["B-ORG"]) : str(self.loader.label2id["I-ORG"]),
                    str(self.loader.label2id["B-PER"]) : str(self.loader.label2id["I-PER"]),
                    str(self.loader.label2id["B-LOC"]) : str(self.loader.label2id["I-LOC"])}
        
        for t in range(len(expr)):
            true_names_idx = [i.start() for i in re.finditer(expr[t], y_true_strings)]
            pred_names_idx = [i.start() for i in re.finditer(expr[t], y_pred_strings)]
            cnt_true_names = len(true_names_idx)
            cnt_pred_names = len(pred_names_idx)
            
            cnt = 0
            t1 = ""
            t2 = ""
            idx_1 = 0
            idx_2 = 0
            i = true_names_idx[idx_1]
            j = pred_names_idx[idx_2]
            while idx_1 < true_names_idx[-1] and idx_2 < pred_names_idx[-1]:
                t1 += y_true_strings[i]
                t2 += y_pred_strings[j]
                
                for k in range(1, len(y_true_strings)):
#                    if y_true_strings[i+1] != str(b_i_dict[int(y_true_strings[i])]) or y_pred_strings[j+1] != str(b_i_dict[int(y_pred_strings[j])]):
                    if y_true_strings[i+1] != b_i_dict[y_true_strings[i]] or y_pred_strings[j+1] != b_i_dict[y_pred_strings[j]]:
                        if y_true_strings[i] == y_pred_strings[j]:
                            cnt += 1
#                        print(y_true_strings[i:i+2], y_pred_strings[j:j+2])
                        break
                    
                    flag_1 = True
                    flag_2 = True
                    
                    t1 += y_true_strings[i+k]
                    if y_true_strings[i+k] == y_true_strings[i+1]:
                        flag_1 = True
                    else:
                        flag_1 = False
                        
                    t2 += y_pred_strings[j+k]
                    if y_pred_strings[j+k] == y_pred_strings[j+1]:
                        flag_2 = True
                    else:
                        flag_2 = False
                    
                    if not (flag_1 and flag_2):
                        break
                
                if t1 == t2:
                    cnt += 1
                
                t1 = ""
                t2 = ""
                
                try:
                    idx_1 += 1
                    idx_2 += 1
                    
                    i = true_names_idx[idx_1]
                    j = pred_names_idx[idx_2]
                    
                    while i != j:
                        if i < j:
                            idx_1 += 1
                            i = true_names_idx[idx_1]
                        else:
                            idx_2 += 1
                            j = pred_names_idx[idx_2]
                except IndexError:
                    break
            
            precision[t] = cnt / cnt_pred_names
            recall[t] = cnt / cnt_true_names
            f1[t] = 2 * precision[t] * recall[t] / (precision[t] + recall[t])

        df = pd.DataFrame(np.array([precision, recall, f1]).T, columns=["Precision", "Recall", "F1"], index=["ORG", "PER", "LOC"])
        pd.set_option("precision", 2)
        print(df)
        fig = df.plot(kind="bar", rot=45).get_figure()
        fig.savefig("scores.png", dpi=300)
        
#        y_true_names = y_true_strings.split(str(self.loader.label2id["O"]))
#        y_true_names = re.split(expr_1, y_true_strings)
#        while None in y_true_names: y_true_names.pop(y_true_names.index(None))
#        while '' in y_true_names: y_true_names.pop(y_true_names.index(''))
#        cnt_true_names = np.sum([i != '' for i in y_true_names])
        
#        y_pred_names = y_pred_strings.split(str(self.loader.label2id["O"]))
#        y_pred_names = re.split(expr_1, y_pred_strings)
#        while None in y_pred_names: y_pred_names.pop(y_pred_names.index(None))
#        while '' in y_pred_names: y_pred_names.pop(y_pred_names.index(''))
#        cnt_pred_names = np.sum([i != '' for i in y_pred_names])
        
#        i = 0
#        j = 0
#        while i < len(y_true_names) and j < len(y_pred_names):
##            while i < len(y_true_names) and y_true_names[i] == '':
##                i += 1
##            while j < len(y_pred_names) and y_pred_names[j] == '':
##                j += 1
#                        
#            if re.match(expr_2, y_true_names[i]) != None:
#                i += 1
#            if re.match(expr_2, y_pred_names[j]) != None:
#                j += 1
#
#            if i >= len(y_true_names) or j >= len(y_pred_names):
#                break
#            
#            if y_true_names[i] == y_pred_names[j]:
#                print(y_true_names[i], y_pred_names[j])
#                cnt += 1
#            
#            if len(y_true_names[i]) > len(y_pred_names[j]):
#                i += 1
#                
#                j += 1 + (len(y_true_names[i]) - len(y_pred_names[j]))
#            elif len(y_true_names[i]) < len(y_pred_names[j]):
#                i += 1 + (len(y_pred_names[j]) - len(y_true_names[i]))
#                j += 1
#            else:
#                i += 1
#                j += 1

#        print(len(y_true_names), len(y_pred_names))

#        y_true = np.array(y_true)
#        real_name_idx = (y_true != self.loader.label2id["O"])
#        real_name = y_true[real_name_idx]
#        
#        y_pred = np.array(y_pred)
#        pred_name_idx = (y_pred != self.loader.label2id["O"])
#        pred_name = y_pred[pred_name_idx]
#        
#        real_names = []
#        temp_real_name = []
#        for i in range(len(real_name_idx)):
#            if real_name_idx[i] == True:

    def savePrediction(self, save_seq):
        with open(self.save_prediction_file, "w") as f:
            for i in range(len(save_seq)):
                t = " ".join(self.loader.id2label[save_seq[i]].tolist()) + "\n"
                f.write(t)

if __name__ == "__main__":
    model = BiLSTM_CRF()
#    model.train()
    model.evaluate()