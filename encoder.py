from __future__ import absolute_import

import importlib
import tensorflow as tf
import numpy as np
import ntm_cell
from ntm_cell import NTMCell
#from ntm import NTM
from collections import defaultdict
from utils import progress

from utils import pp

class videoCaption():
    def __init__(self,cell,dim_image,batch_size, n_frame_step, min_length, max_length, controller_layer_size, dim_hidden,
                 write_head_size, read_head_size, n_words, scope, lr=1e-4, momentum=0.9,min_grad=-10, max_grad=+10,
                 decay=0.95, drop_out_rate=.5, bias_init_vector=None):
        if not isinstance(cell, ntm_cell.NTMCell):
            raise TypeError("cell must be an instance of NTMCell")

        self.cell=cell
        self.dim_image=dim_image
        self.batch_size = batch_size
        self.n_frame_step = n_frame_step
        self.min_length=min_length
        self.max_length=max_length
        self.controller_layer_size=controller_layer_size
        self.dim_hidden=dim_hidden
        self.write_head_size=write_head_size
        self.read_head_size=read_head_size
        self.n_words = n_words
        self.scope=scope
        self.lr=lr
        self.momentum=momentum
        self.min_grad = min_grad
        self.max_grad = max_grad
        self.decay=decay
        self.drop_out_rate = drop_out_rate

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')
        self.embed_att_w = tf.Variable(tf.random_uniform([dim_hidden, 1], -0.1, 0.1), name='embed_att_w')
        self.embed_att_Wa = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1, 0.1), name='embed_att_Wa')
        self.embed_att_Ua = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1, 0.1), name='embed_att_Ua')
        self.embed_att_ba = tf.Variable(tf.zeros([dim_hidden]), name='embed_att_ba')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        self.embed_nn_Wp = tf.Variable(tf.random_uniform([3 * dim_hidden, dim_hidden], -0.1, 0.1), name='embed_nn_Wp')
        self.embed_nn_bp = tf.Variable(tf.zeros([dim_hidden]), name='embed_nn_bp')

        self.inputs = []
        self.outputs = {}
        self.output_logits = {}
        self.true_outputs = []

        self.prev_states = {}
        self.input_states = defaultdict(list)
        self.output_states = defaultdict(list)

        self.losses = {}
        self.optims = {}
        self.grads = {}

        self.saver = None
        self.params = None

        with tf.variable_scope(self.scope):
            self.global_step = tf.Variable(0, trainable=False)


        #self.build_model(forward_only)
    def build_model(self):
        print(" [*] Building a NTM Encoder")
        with tf.variable_scope(self.scope):
            zeros = np.zeros(self.cell.input_dim, dtype=np.float32)

            input_ = tf.placeholder(tf.float32, [self.batch_size, self.cell.input_dim],
                                name='input_%s' % 1)
            self.inputs.append(input_)
            prev_state = self.cell(input_, state=None)
            tf.get_variable_scope().reuse_variables()
            for seq_length in range(2, self.max_length + 1):
            #print(seq_length)
                progress(seq_length / float(self.max_length))

                input_ = tf.placeholder(tf.float32, [self.batch_size, self.cell.input_dim],
                                    name='input_%s' % seq_length)                           # b*d
                self.inputs.append(input_)

                # present inputs
                prev_state = self.cell(input_, prev_state)


            print("Getting attention weighted features")
            video = tf.placeholder(tf.float32, [self.batch_size, self.n_frame_step, self.dim_image])  # b x n x d
            video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frame_step])  # b x n

            caption = tf.placeholder(tf.int32, [self.batch_size, n_caption_step])  # b x c
            caption_mask = tf.placeholder(tf.float32, [self.batch_size, n_caption_step])  # b x c


            video_flat = tf.reshape(video, [-1, self.dim_image])  # (b x n) x d
            image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)  # (b x n) x h
            image_emb = tf.reshape(image_emb, [self.batch_size, self.n_frame_step, self.dim_hidden])  # b x n x h
            image_emb = tf.transpose(image_emb, [1, 0, 2])  # n x b x h

            loss_caption = 0.0

            h_prev = tf.zeros([self.batch_size, self.dim_hidden])  # b x h
            generated_words = []

            current_embed = tf.zeros([self.batch_size, self.dim_hidden])  # b x h
            brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_frame_step, 1, 1])  # n x h x 1
            image_part = tf.matmul(image_emb, tf.tile(tf.expand_dims(self.embed_att_Ua, 0),
                                                        [self.n_frame_step, 1, 1])) + self.embed_att_ba  # n x b x h

            for i in range(n_caption_step):
                e = tf.tanh(tf.matmul(h_prev, self.embed_att_Wa) + image_part)  # n x b x h
                e = tf.matmul(e, brcst_w)  # unnormalized relevance score
                e = tf.reduce_sum(e, 2)  # n x b
                e_hat_exp = tf.multiply(tf.transpose(video_mask), tf.exp(e))  # n x b
                denomin = tf.reduce_sum(e_hat_exp, 0)  # b
                denomin = denomin + tf.to_float(tf.equal(denomin, 0))  # regularize denominator
                alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp, denomin), 2),
                             [1, 1, self.dim_hidden])  # n x b x h  # normalize to obtain alpha
                attention_list = tf.multiply(alphas, image_emb)  # n x b x h
                atten = tf.reduce_sum(attention_list, 0)  # b x h       #  soft-attention weighted sum
                if i > 0: tf.get_variable_scope().reuse_variables()
                prev_state = self.cell(tf.concat([atten, current_embed],1), prev_state)
                output1=tf.squeeze(prev_state['output'])

                output2 = tf.tanh(tf.nn.xw_plus_b(tf.concat([output1, atten, current_embed],1), self.embed_nn_Wp,
                                                  self.embed_nn_bp))  # b x h

                h_prev = output1  # b x h
                labels = tf.expand_dims(caption[:, i], 1)  # b x 1
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)  # b x 1
                concated = tf.concat([indices, labels],1)  # b x 2
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0,
                                                   0.0)  # b x w
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)  # b x w
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)  # b x 1
                cross_entropy = cross_entropy * caption_mask[:, i]  # b x 1
                loss_caption += tf.reduce_sum(cross_entropy)  # 1

            loss_caption = loss_caption / tf.reduce_sum(caption_mask)
            loss = loss_caption
            return loss, video, video_mask, caption, caption_mask

from types import FunctionType
def methods(cls):
    return [x for x, y in cls.__dict__.items() if type(y) == FunctionType]
############## Train Parameters #################
dim_image = 10	# same as input_dim
dim_hidden= 5		# same as controller_dim
n_frame_step = 2   #same as num_images
batch_size = 4
min_length=1
max_length=2
controller_layer_size=1
write_head_size=1
read_head_size=1
output_dim=10
n_caption_step=3
##################################################

def train():
    cell= NTMCell(input_dim=dim_image,mem_size=2, mem_dim=7, controller_dim=dim_hidden,controller_layer_size=1,
                  shift_range=1, write_head_size=1, read_head_size=1, batch_size=batch_size)
    model=videoCaption(cell=cell, dim_image=dim_image,batch_size=batch_size,n_frame_step=n_frame_step,min_length=min_length
                       ,max_length=max_length, controller_layer_size=controller_layer_size, dim_hidden=dim_hidden,
                       write_head_size=write_head_size, read_head_size=read_head_size,n_words=100,scope='NTM')
    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask= model.build_model()



train()

