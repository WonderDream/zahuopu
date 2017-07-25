# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:38:04 2017

@author: zhujisong-001
"""
import re
import jieba

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os

os.chdir('E:/Data/tensorflow')

fpoems = 'E:/Data/poems.txt'

with open(fpoems, "r", encoding = 'gb2312') as f:
        poems = f.read()

poems = re.sub('\n{6}.*\n{6}', ' ', poems)
poemsByPoet = re.split('【.*】', poems)

jiebaIMM = False
words = set(jieba.cut(poems, HMM = jiebaIMM))
word2id = dict(zip(words,  3 + np.array(range(len(words))))) #reserve 3 ids for PAD_ID, START_ID, END_ID

PAD_ID = 0
START_ID = 1
END_ID = 2

UNK = '_UNK'
word2id[UNK] = max(word2id.values()) + 1
id2word = dict(zip(word2id.values(), word2id.keys()))
id2word[2] = '。'

def wordToId(word):
    return(word2id.get(word, UNK))
def idToWord(wordId):
    return(id2word.get(wordId, UNK))
    
vocabSize = max(word2id.values())
embeddingSize = 500
max_grad_norm = 5
max_epoch = 4
learning_rate = 0.9
lr_decay = 0.9

batch_size = 20
input_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="input_seqs")
decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
output_seqs = tf.placeholder(tf.int64, [batch_size, None], name = 'output_seqs')
target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask")

input_seq_test = tf.placeholder(tf.int64, [1, None], name = 'input_seq_test')
decode_seq_test = tf.placeholder(tf.int64, [1, None], name = 'decode_seq_test')

def inferenceNet(x, xdecode, reuse=None):
    
    with tf.variable_scope("model", reuse=reuse):
        
        with tf.variable_scope("embedding") as vs:
            net_encode = tl.layers.EmbeddingInputlayer(
                    inputs = x,
                    vocabulary_size = vocabSize,
                    embedding_size = embeddingSize,
                    name = 'seq_embedding')
            vs.reuse_variables()
            tl.layers.set_name_reuse(True)
            net_decode = tl.layers.EmbeddingInputlayer(
                    inputs = xdecode,
                    vocabulary_size = vocabSize,
                    embedding_size = embeddingSize,
                    name = 'seq_embedding')
            
        tl.layers.set_name_reuse(reuse)
        network = tl.layers.Seq2Seq(net_encode, net_decode,
                                    cell_fn = tf.contrib.rnn.BasicLSTMCell, # for TF0.2 tf.nn.rnn_cell.BasicLSTMCell,
                                    n_hidden = embeddingSize,
                                    dropout = 0.7,
                                    encode_sequence_length = tl.layers.retrieve_seq_length_op2(x),
                                    decode_sequence_length = tl.layers.retrieve_seq_length_op2(xdecode),
                                    return_seq_2d = True,     # stack denselayer or compute cost after it
                                    n_layer = 1,
                                    name = 'seq2seq')
        seq2seqLayer = network
        network = tl.layers.DenseLayer(network, n_units=vocabSize, act=tf.identity, name="output")
    return seq2seqLayer, network

seq2seqLayer, network = inferenceNet(input_seqs, decode_seqs, reuse = None)
seq2seqLayerTest, networkTest = inferenceNet(input_seq_test, decode_seq_test, reuse = True)
y_linear = networkTest.outputs
y_soft = tf.nn.softmax(y_linear)

loss = tl.cost.cross_entropy_seq_with_mask(
            network.outputs,
            output_seqs,
            target_mask)
cost = tf.reduce_sum(loss) / batch_size
                    
tvars = network.all_params
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)

with tf.variable_scope('learning_rate'):
        lr = tf.Variable(0, trainable=False, dtype = tf.float32)
        
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.apply_gradients(zip(grads, tvars))



inputLines = []
outputLines = []
for i in range(len(poemsByPoet)):

    batch = poemsByPoet[i].split()
    if len(batch) == 0:
        continue
    
    segedLines = [[wordToId(word) for word in jieba.cut(line, HMM = jiebaIMM)] for line in batch]
    
    previousLines = segedLines[0:(len(segedLines) - 1)]
    nextLines = segedLines[1:len(segedLines)]
    
    inputLines.extend(previousLines)
    outputLines.extend(nextLines)

maxInputLen = max([len(line) for line in inputLines])
paddedInputLines = [line + [PAD_ID]*(maxInputLen - len(line)) for line in inputLines]
decodeSeqs = [[START_ID] + line for line in paddedInputLines]

outputLines = [line + [END_ID] for line in outputLines]
maxOutputLen = max([len(line) for line in outputLines])
paddedOutputLines = [line + [PAD_ID]*(maxOutputLen - len(line)) for line in outputLines]
outputMask = [[1]*len(line) + [0]*(maxOutputLen - len(line)) for line in outputLines]

batchCnt = len(paddedInputLines)//batch_size

saver = tf.train.Saver()

sess = tf.Session()
tl.layers.initialize_global_variables(sess)

ckptDir = './my-model'
initial_step = 0
ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckptDir))
if ckpt and ckpt.model_checkpoint_path:
# Restores from checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)
    initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
for i in range(initial_step, batchCnt):
    
    if i % 100 == 0:
        saver.save(sess, ckptDir, global_step=i)
    
    new_lr_decay = lr_decay ** max(i - max_epoch, 0.0)
    sess.run(tf.assign(lr, learning_rate * new_lr_decay))
   
    batchInput = np.array(paddedInputLines[i*batch_size : (i + 1)*batch_size])
    batchDecode = np.array(decodeSeqs[i*batch_size : (i + 1)*batch_size])
    batchOutput = np.array(paddedOutputLines[i*batch_size : (i + 1)*batch_size])
    batchMask = np.array(outputMask[i*batch_size : (i + 1)*batch_size])
    
    _cost, state1, _ = sess.run([cost,
                                    seq2seqLayer.final_state,
                                    train_op],
                                    feed_dict={input_seqs: batchInput, decode_seqs:batchDecode, output_seqs: batchOutput, target_mask : batchMask,})
        
    

testStartSeq = '夏天的风，'
testSeqId = [[wordToId(word) for word in jieba.cut(testStartSeq, HMM = jiebaIMM)]]
decodeTest = [[START_ID] + line for line in testSeqId]


dp_dict = tl.utils.dict_to_one(networkTest.all_drop )

y_id = None
for i in range(1000):
    if i > 0:
        testSeqId = [y_id]
        decodeTest = [[START_ID] + line for line in testSeqId]
    feed_dict={input_seq_test:testSeqId,decode_seq_test:decodeTest}
    feed_dict.update(dp_dict)
    res = sess.run(y_soft, feed_dict=feed_dict)
    y_id = np.argmax(res, 1)
    print([idToWord(wid) for wid in y_id])
    

