#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 08:47:52 2018

@author: congxin
"""

#from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
import random
import pickle
import os
import pdb

class DataLoader(object):
    def __init__(self, reload=False):
        self.src_data_file = "data/source.txt"
        self.src_label_file = "data/target.txt"
        self.test_data_file = "data/test.txt"
        self.test_label_file = "data/test_tgt.txt"
        self.dict_file = "data/dict.pkl"
        self.src_file = "data/source.pkl"
        self.test_file = "data/test.pkl"
        
        if not reload and os.path.exists(self.dict_file):
            self.loadDicts()
        else:
            self.createDicts()
            
        if not reload and os.path.exists(self.src_file):
            self.loadTrainData()
        else:
            self.createTrainData()
            
        if not reload and os.path.exists(self.test_file):
            self.loadTestData()
        else:
            self.createTestData()
            
        self.vocab_size = len(self.word2id)
        self.label_size = len(self.label2id)
        self.train_size = len(self.sentences)
        self.test_size = len(self.test_sentences)
    
    def createDicts(self):
        with open(self.src_data_file, "r") as f1, open(self.src_label_file, "r") as f2:
            words = f1.read().strip().split()
            labels = f2.read().strip().split()
        
        self.word_dict = list(set(words))
        self.label_dict = list(set(labels))
        
        self.words_ids = range(1, len(self.word_dict)+1)
        self.labels_ids = range(0, len(self.label_dict)) # prediction of bilstm-crf in [0, 6] !!!
        self.word2id = pd.Series(self.words_ids, index=self.word_dict)
        self.id2word = pd.Series(self.word_dict, index=self.words_ids)
        self.label2id = pd.Series(self.labels_ids, index=self.label_dict)
        self.id2label = pd.Series(self.label_dict, index=self.labels_ids)
        print("Dict created")
        
        with open(self.dict_file, "wb") as f:
            pickle.dump(self.word_dict, f)
            pickle.dump(self.label_dict, f)
            pickle.dump(self.word2id, f)
            pickle.dump(self.id2word, f)
            pickle.dump(self.label2id, f)
            pickle.dump(self.id2label, f)
        print("Saved dict")
    
    def loadDicts(self):
        with open(self.dict_file, "rb") as f:
            self.word_dict = pickle.load(f)
            self.label_dict = pickle.load(f)
            self.word2id = pickle.load(f)
            self.id2word = pickle.load(f)
            self.label2id = pickle.load(f)
            self.id2label = pickle.load(f)
    
    def createTrainData(self):
        with open(self.src_data_file, "r") as f1, open(self.src_label_file, "r") as f2:
            self.sentences = f1.readlines()
            self.labels = f2.readlines()
            self.sentences = [i.strip().split() for i in self.sentences]
            self.labels = [i.strip().split() for i in self.labels]
        print("Loaded training data")
        
        self.id_sentences = []
        self.id_labels = []
        self.sentences_lengths = []
        
        print("Begin to preprocess test data")
        for i in range(len(self.sentences)):
            temp_sentence = list(self.word2id[self.sentences[i]])
            temp_label = list(self.label2id[self.labels[i]])
            self.id_sentences.append(temp_sentence)
            self.id_labels.append(temp_label)
            self.sentences_lengths.append(len(temp_sentence))
            print("Preprocessed training sentence {}".format(i))
        
        with open(self.src_file, "wb") as f:
            pickle.dump(self.sentences, f)
            pickle.dump(self.labels, f)
            pickle.dump(self.id_sentences, f)
            pickle.dump(self.id_labels, f)
            pickle.dump(self.sentences_lengths, f)
        print("Saved training data")
        
    def loadTrainData(self):
        with open(self.src_file, "rb") as f:
            self.sentences = pickle.load(f)
            self.labels = pickle.load(f)
            self.id_sentences = pickle.load(f)
            self.id_labels = pickle.load(f)
            self.sentences_lengths = pickle.load(f)
    
    ## verify each length
    #max_len = 0
    #for i in range(len(sentences)):
    #    if len(sentences[i]) != len(seq_labels[i]):
    #        print("No. %d is fault!" % i)
    #    if len(sentences[i]) > max_len:
    #            max_len = len(sentences[i])
    ##print("max length = %d" % max_len)
    #padding_length = max_len
    
    ## encode into one-hot
    #encoder = LabelBinarizer()
    #tokens = encoder.fit_transform(words)
    
    def createTestData(self):
        with open(self.test_data_file, "r") as f1, open(self.test_label_file, "r") as f2:
            self.test_sentences = f1.readlines()
            self.test_labels = f2.readlines()
            self.test_sentences = [i.strip().split() for i in self.test_sentences]
            self.test_labels = [i.strip().split() for i in self.test_labels]
        print("Loaded test data")
        
        self.id_test_sentences = []
        self.id_test_labels = []
        self.test_sentences_lengths = []
        
        print("Begin to preprocess test data")
        for i in range(len(self.test_sentences)):
            temp_sentence = list(self.word2id[self.test_sentences[i]].fillna(0)) # testset may contain unkonwn characters
            temp_label = list(self.label2id[self.test_labels[i]])
            self.id_test_sentences.append(temp_sentence)
            self.id_test_labels.append(temp_label)
            self.test_sentences_lengths.append(len(temp_sentence))
            print("Preprocessed test sentence {}".format(i))
        
        with open(self.test_file, "wb") as f:
            pickle.dump(self.test_sentences, f)
            pickle.dump(self.test_labels, f)
            pickle.dump(self.id_test_sentences, f)
            pickle.dump(self.id_test_labels, f)
            pickle.dump(self.test_sentences_lengths, f)
        print("Saved test data")
        
    def loadTestData(self):
        with open(self.test_file, "rb") as f:
            self.test_sentences = pickle.load(f)
            self.test_labels = pickle.load(f)
            self.id_test_sentences = pickle.load(f)
            self.id_test_labels = pickle.load(f)
            self.test_sentences_lengths = pickle.load(f)
            
    def padSentences(self, batch):
        max_len = max(map(lambda x : len(x), batch[0]))
        new_sentences = []
        new_labels = []
        for sentence, label in zip(batch[0], batch[1]):
            sentence.extend([0] * (max_len - len(sentence)))
            label.extend([self.label2id['O']] * (max_len - len(label)))
            new_sentences.append(sentence)
            new_labels.append(label)
        new_batch = (new_sentences, new_labels, batch[2])
        return new_batch
            
    def getTrainBatch(self, batch_size):
        offset = random.sample(range(self.train_size), batch_size)
        batch_sentences = []
        batch_labels = []
        batch_sentences_lengths = []
        for i in offset:
            batch_sentences.append(self.id_sentences[i])
            batch_labels.append(self.id_labels[i])
            batch_sentences_lengths.append(self.sentences_lengths[i])
        return self.padSentences((batch_sentences, batch_labels, batch_sentences_lengths))
    
    def getTestBatch(self, batch_size):
        offset = random.sample(range(self.test_size), batch_size)
        batch_sentences = []
        batch_labels = []
        batch_sentences_lengths = []
        for i in offset:
            batch_sentences.append(self.id_test_sentences[i])
            batch_labels.append(self.id_test_labels[i])
            batch_sentences_lengths.append(self.test_sentences_lengths[i])
        return self.padSentences((batch_sentences, batch_labels, batch_sentences_lengths))
    
    def getAllTestData(self):
        return self.padSentences((self.id_test_sentences, self.id_test_labels, self.test_sentences_lengths))
    

if __name__ == "__main__":
    t = DataLoader(reload=True)
    a = t.getTrainBatch(12)
    b = t.getTestBatch(3)
    c = t.getAllTestData()