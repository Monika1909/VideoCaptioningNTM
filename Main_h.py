from __future__ import absolute_import
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import importlib
import tensorflow as tf
import numpy as np
import ntm_cell
from ntm_cell import NTMCell
import pandas as pd
#from ntm import NTM
from collections import defaultdict
from utils import progress
import h5py, json
from keras.preprocessing import sequence
import os
from utils import pp
import unicodedata
import sys
from cocoeval import COCOScorer
from nltk.translate.bleu_score import corpus_bleu
tf.set_random_seed(1234)
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

        self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')
        self.embed_att_w = tf.Variable(tf.random_uniform([dim_hidden, 1], -0.1, 0.1), name='embed_att_w')
        self.embed_att_Wa = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1, 0.1), name='embed_att_Wa')
        self.embed_att_Ua = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1, 0.1), name='embed_att_Ua')
        self.embed_att_ba = tf.Variable(tf.zeros([dim_hidden]), name='embed_att_ba')
        self.W_beta_1 = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1, 0.1), name='W_beta_1')
        self.b_beta_1 = tf.Variable(tf.zeros([dim_hidden]), name='b_beta_1')
        self.W_beta_2 = tf.Variable(tf.random_uniform([dim_hidden, cell.mem_dim], -0.1, 0.1), name='W_beta_2')
        self.b_beta_2 = tf.Variable(tf.zeros([cell.mem_dim]), name='b_beta_2')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        self.embed_nn_Wp = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1, 0.1), name='embed_nn_Wp')
        self.embed_nn_Up = tf.Variable(tf.random_uniform([dim_hidden + dim_hidden + self.cell.mem_dim, dim_hidden], -0.1, 0.1), name='embed_nn_Up')
        self.embed_nn_bp = tf.Variable(tf.zeros([dim_hidden]), name='embed_nn_bp')


        self.saver = None

        with tf.variable_scope(self.scope):
            self.global_step = tf.Variable(0, trainable=False)

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_frame_step, self.dim_image])  # b x n x d
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frame_step])  # b x n

        caption = tf.placeholder(tf.int32, [self.batch_size, n_caption_step])  # b x c
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, n_caption_step])  # b x c

        video_flat = tf.reshape(video, [-1, self.dim_image])  # (b x n) x d
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)  # (b x n) x h
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_frame_step, self.dim_hidden])  # b x n x h
        image_emb = tf.transpose(image_emb, [1, 0, 2])  # n x b x h

        print(" [*] Building a NTM Encoder")
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_ = tf.squeeze(image_emb[0, :, :])

            beta2 = tf.ones([self.batch_size, self.cell.mem_dim])
            prev_state = self.cell(input_, beta2, state=None, isEncoder=True, use=False)
            tf.get_variable_scope().reuse_variables()
            for seq_length in range(2, self.max_length + 1):
                progress(seq_length / float(self.max_length))
                input_ = tf.squeeze(image_emb[seq_length-1, :, :])

                prev_state = self.cell(input_, beta2, prev_state)

            print("Getting attention weighted features")

            loss_caption = 0.0
            h_prev = tf.squeeze(prev_state['output'])           #b x h

            current_embed = tf.zeros([self.batch_size, self.dim_hidden])  # b x h
            image_part = tf.matmul(image_emb, tf.tile(tf.expand_dims(self.embed_att_Ua, 0),
                                                      [self.n_frame_step, 1, 1])) + self.embed_att_ba  # n x b x d
            brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_frame_step, 1, 1])  # n x d x 1
            for i in range(n_caption_step):
                Beta_1=tf.sigmoid(tf.matmul(h_prev, self.W_beta_1) + self.b_beta_1)
                Beta_2 = tf.sigmoid(tf.matmul(h_prev, self.W_beta_2) + self.b_beta_2)
                e = tf.tanh(tf.matmul(h_prev, self.embed_att_Wa) + image_part)  # n x b x d
                e = tf.matmul(e, brcst_w)  # unnormalized relevance score
                e = tf.reduce_sum(e, 2)  # n x b
                e_hat_exp = tf.multiply(tf.transpose(video_mask), tf.exp(e))  # n x b
                denomin = tf.reduce_sum(e_hat_exp, 0)  # b
                denomin = denomin + tf.to_float(tf.equal(denomin, 0))  # regularize denominator
                alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp, denomin), 2),
                             [1, 1, self.dim_hidden])  # n x b x d  # normalize to obtain alpha
                attention_list = tf.multiply(alphas, image_emb)  # n x b x d
                atten = tf.reduce_sum(attention_list, 0)  # b x d       #  soft-attention weighted sum
                if i > 0: tf.get_variable_scope().reuse_variables()
                prev_state = self.cell(tf.concat([tf.multiply(Beta_1,atten), current_embed],1),Beta_2, prev_state, isEncoder=False)
                output1=tf.squeeze(prev_state['output'])
                zmt=tf.squeeze(prev_state['read'])

                #output2 = tf.tanh(tf.nn.xw_plus_b(tf.concat([output1, atten, current_embed, zmt],1), self.embed_nn_Wp,
                                                  #self.embed_nn_bp))  # b x h

                output2 = tf.matmul(tf.tanh(current_embed), self.embed_nn_Wp) + tf.matmul(tf.concat([output1, atten, zmt], 1), self.embed_nn_Up) + self.embed_nn_bp
                h_prev = output1  # b x h
                labels = tf.expand_dims(caption[:, i], 1)  # b x 1
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)  # b x 1
                concated = tf.concat([indices, labels],1)  # b x 2
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0,
                                                   0.0)  # b x w
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)  # b x w
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_words, labels=onehot_labels)  # b x 1
                cross_entropy = cross_entropy * caption_mask[:, i]  # b x 1
                loss_caption += tf.reduce_sum(cross_entropy)  # 1

            loss_caption = loss_caption / tf.reduce_sum(caption_mask)
            loss = loss_caption
        return loss, video, video_mask, caption, caption_mask

    def build_generator(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_frame_step, self.dim_image])  # b x n x d
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frame_step])  # b x n

        video_flat = tf.reshape(video, [-1, self.dim_image])  # (b x n) x d
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)  # (b x n) x h
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_frame_step, self.dim_hidden])  # b x n x h
        image_emb = tf.transpose(image_emb, [1, 0, 2])  # n x b x h
        print(" [*] Building a NTM Encoder")
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_ = tf.squeeze(image_emb[0, :, :])
            beta2 = tf.ones([self.batch_size, self.cell.mem_dim])
            tf.get_variable_scope().reuse_variables()
            prev_state_g = self.cell(input_,beta2, state=None, isEncoder=True, use=False)
            for seq_length in range(2, self.max_length + 1):
            #print(seq_length)
                progress(seq_length / float(self.max_length))

                #input_ = tf.placeholder(tf.float32, [self.batch_size, self.cell.input_dim], name='input_%s' % seq_length)                           # b*d
                #input_ = tf.Variable(video[:, seq_length-1, :], trainable=False, name='input_%s' % seq_length)
                input_ = tf.squeeze(image_emb[seq_length-1, :, :])

                # present inputs
                prev_state_g = self.cell(input_,beta2, prev_state_g)

            print("Getting attention weighted features")


            generated_words = []
            h_prev = tf.squeeze(prev_state_g['output'])  # b x h

            current_embed = tf.zeros([self.batch_size, self.dim_hidden])  # b x h
            image_part = tf.matmul(image_emb, tf.tile(tf.expand_dims(self.embed_att_Ua, 0),
                                                                           [self.n_frame_step, 1,
                                                                            1])) + self.embed_att_ba  # n x b x d
            brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_frame_step, 1, 1])  # n x d x 1
            for i in range(n_caption_step):
                Beta_1=tf.sigmoid(tf.matmul(h_prev, self.W_beta_1) + self.b_beta_1)
                Beta_2 = tf.sigmoid(tf.matmul(h_prev, self.W_beta_2) + self.b_beta_2)
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
                prev_state_g = self.cell(tf.concat([tf.multiply(Beta_1,atten), current_embed],1),Beta_2, prev_state_g, isEncoder=False)
                output1=tf.squeeze(prev_state_g['output'])
                zmt=tf.squeeze(prev_state_g['read'])
                output2 = tf.matmul(tf.tanh(current_embed), self.embed_nn_Wp) + tf.matmul(tf.concat([output1, atten, zmt], 1), self.embed_nn_Up) + self.embed_nn_bp

                h_prev = output1  # b x h

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)  # b x w
                max_prob_index = tf.argmax(logit_words, 1)  # b
                generated_words.append(max_prob_index)  # b
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)

            generated_words = tf.transpose(tf.stack(generated_words))
        return video, video_mask, generated_words


def get_video_data_HL(video_data_path, video_feat_path):
    files = open(video_data_path)
    List = []
    for ele in files:
        List.append(ele[:-1])
    return np.array(List)

def get_video_data_jukin(video_data_path_train, video_data_path_val, video_data_path_test):
    video_list_train = get_video_data_HL(video_data_path_train, video_feat_path)
    train_title = []
    title = []
    fname = []
    for ele in video_list_train:
        batch_data = h5py.File(ele)
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in range(len(batch_fname)):
            fname.append(batch_fname[i])
            title.append(batch_title[i])
            train_title.append(batch_title[i])

    video_list_val = get_video_data_HL(video_data_path_val, video_feat_path)
    for ele in video_list_val:
        batch_data = h5py.File(ele)
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in range(len(batch_fname)):
            fname.append(batch_fname[i])
            title.append(batch_title[i])
            #train_title.append(batch_title[i])

    video_list_test = get_video_data_HL(video_data_path_test, video_feat_path)
    for ele in video_list_test:
        batch_data = h5py.File(ele)
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in range(len(batch_fname)):
            fname.append(batch_fname[i])
            title.append(batch_title[i])


    fname = np.array(fname)
    title = np.array(title)
    train_title = np.array(train_title)
    video_data = pd.DataFrame({'Description': train_title})

    return video_data, video_list_train, video_list_val, video_list_test

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):  # borrowed this function from NeuralTalk
    print('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0  # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector)  # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)  # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

def testing_one(sess, video_feat_path, ixtoword, video_tf, video_mask_tf, caption_tf, counter):
    pred_sent = []
    gt_sent = []
    IDs = []
    namelist = []
    # print video_feat_path
    test_data_batch = h5py.File(video_feat_path)
    gt_captions = json.load(open('msvd2sent.json'))

    video_feat = np.zeros((batch_size, n_frame_step, dim_image))
    video_mask = np.zeros((batch_size, n_frame_step))
    #    video_feat = np.transpose(test_data_batch['data'],[1,0,2])
    findZeros=np.zeros(dim_image)
    for ind in range(batch_size):
        video_feat[ind, :, :] = test_data_batch['data'][:n_frame_step, ind, :]
        idx=np.where((test_data_batch['data'][:, ind] == findZeros).all(axis=1))[0]
        if (len(idx) == 0):
            continue
        video_mask[ind, :idx[-1] + 1] = 1.

    generated_word_index = sess.run(caption_tf, feed_dict={video_tf: video_feat, video_mask_tf: video_mask})
    # ipdb.set_trace()

    for ind in range(batch_size):
        cap_key = test_data_batch['fname'][ind].decode('unicode-escape')
        if cap_key == '':
            break
        else:
            generated_words = ixtoword[generated_word_index[ind]]
            punctuation = np.argmax(np.array(generated_words) == '.') + 1
            generated_words = generated_words[:punctuation]
            # ipdb.set_trace()
            generated_sentence = ' '.join(generated_words)
            pred_sent.append([{'image_id': str(counter), 'caption': generated_sentence}])
            namelist.append(cap_key)
            for i, s in enumerate(gt_captions[cap_key]):
                s = unicodedata.normalize('NFKD', s)
                gt_sent.append([{'image_id': str(counter), 'cap_id': i, 'caption': s}])
                IDs.append(str(counter))
            counter += 1

    return pred_sent, gt_sent, IDs, counter, namelist


def testing_all(sess, test_data, ixtoword, video_tf, video_mask_tf, caption_tf):
    pred_sent = []
    gt_sent = []
    IDs_list = []
    flist = []
    counter = 0
    gt_dict = defaultdict(list)
    pred_dict = {}
    for _, video_feat_path in enumerate(test_data):
        [b, c, d, counter, fns] = testing_one(sess, video_feat_path, ixtoword, video_tf, video_mask_tf, caption_tf,
                                              counter)
        pred_sent += b
        gt_sent += c
        IDs_list += d
        flist += fns

    for k, v in zip(IDs_list, gt_sent):
        gt_dict[k].append(v[0])

    new_flist = []
    new_IDs_list = []
    for k, v in zip(range(len(pred_sent)), pred_sent):
        if flist[k] not in new_flist:
            new_flist.append(flist[k])
            new_IDs_list.append(str(k))
            pred_dict[str(k)] = v

    # pdb.set_trace()
    return pred_sent, gt_sent, new_IDs_list, gt_dict, pred_dict


############## Train Parameters #################
dim_image = 1024	# same as input_dim
dim_hidden= 512		# same as controller_dim
n_frame_step = 28   #same as num_images
batch_size = 64
min_length=1
max_length=28
controller_layer_size=1
write_head_size=1
read_head_size=1
output_dim=20
n_caption_step=20
learning_rate = 0.0001
n_epochs=50

video_data_path_train = 'h5py/cont_captions/train.txt'
video_data_path_val = 'h5py/cont_captions/val.txt'
video_data_path_test = 'h5py/cont_captions/test.txt'
video_feat_path = 'h5py/cont_augment'
model_path = 'Att_models'

##################################################

def train():
    meta_data, train_data, val_data, test_data = get_video_data_jukin(video_data_path_train, video_data_path_val, video_data_path_test)
    captions = meta_data['Description'].values
    for i in range(captions.size):
        captions[i]=captions[i].decode('unicode-escape')
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=1)

    np.save('data0/ixtoword', ixtoword)
    cell = NTMCell(input_dim=dim_image, mem_size=10, mem_dim=1000, controller_dim=dim_hidden, controller_layer_size=1,
                   shift_range=1, write_head_size=1, read_head_size=1, batch_size=batch_size)
    model=videoCaption(cell=cell, dim_image=dim_image,batch_size=batch_size,n_frame_step=n_frame_step,min_length=min_length
                       ,max_length=max_length, controller_layer_size=controller_layer_size, dim_hidden=dim_hidden,
                       write_head_size=write_head_size, read_head_size=read_head_size, n_words=len(wordtoix), scope='NTM')
    
    #print(tf.contrib.framework.get_name_scope())
    
    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask= model.build_model()
    #print(tf.global_variables())
    #config = tf.ConfigProto(allow_soft_placement=True)
    config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #config.log_device_placement=True
    sess = tf.InteractiveSession(config=config)

    saver = tf.train.Saver(max_to_keep=100)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    #tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()

    for epoch in range(n_epochs):
        print("epoch %d"%epoch)
        index = np.arange(len(train_data))
        np.random.shuffle(index)
        train_data = train_data[index]
        loss_epoch = np.zeros(len(train_data))
        for current_batch_file_idx in range(len(train_data)):
            #print(current_batch_file_idx)
            current_batch = h5py.File(train_data[current_batch_file_idx])
            current_feats = np.zeros((batch_size, n_frame_step, dim_image))                 #b*28*d
            current_video_masks = np.zeros((batch_size, n_frame_step))                         #b*28
            current_video_len = np.zeros(batch_size)
            findZeros=np.zeros(dim_image)
            for ind in range(batch_size):
                current_feats[ind, :, :] = current_batch['data'][:n_frame_step, ind, :]
                #idx = np.where(current_batch['data'][:, ind] == findZeros)[0]
                idx=np.where((current_batch['data'][:, ind] == findZeros).all(axis=1))[0]
                if len(idx) == 0:
                    continue
                current_video_masks[ind, :idx[0]] = 1

            current_captions = list(current_batch['title'])
            for i in range(np.size(current_captions)):
                current_captions[i] = current_captions[i].decode('unicode-escape')
            current_caption_ind = list(map(
                lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions))

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post',
                                                            maxlen=n_caption_step - 1)
            current_caption_matrix = np.hstack(
                [current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int)
            current_caption_masks = np.zeros((batch_size, current_caption_matrix.shape[1]))
            nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix)))
            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1
                
            #print(tf.global_variables())

            _, loss_val = sess.run(
                [train_op, tf_loss],
                feed_dict={
                    tf_video: current_feats,
                    tf_video_mask: current_video_masks,
                    tf_caption: current_caption_matrix,
                    tf_caption_mask: current_caption_masks,
                })
            loss_epoch[current_batch_file_idx] = loss_val
        # print "Time Cost:", round(tStop - tStart,2), "s"

        print("Epoch:", epoch, " done. Loss:", np.mean(loss_epoch))

        if np.mod(epoch, 1) == 0 or epoch == n_epochs - 1:
            #print("Epoch ", epoch, " is done. Saving the model ...")
            #saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

            current_batch = h5py.File(val_data[np.random.randint(0, len(val_data))])
            video_tf, video_mask_tf, caption_tf= model.build_generator()
            ixtoword = pd.Series(np.load('data0/ixtoword.npy').tolist())
            '''[pred_sent, gt_sent, id_list, gt_dict, pred_dict] = testing_all(sess, train_data[-2:], ixtoword, video_tf,
                                                                            video_mask_tf, caption_tf)'''
            
            [pred_sent, gt_sent, id_list, gt_dict, pred_dict] = testing_all(sess, val_data, ixtoword, video_tf,
                                                                            video_mask_tf, caption_tf)

            [pred_sent_test, gt_sent_test, id_list_test, gt_dict_test, pred_dict_test] = testing_all(sess, test_data, ixtoword, video_tf,
                                                                            video_mask_tf, caption_tf)
           
            references=[]
            predictions=[]
            for key in pred_dict.keys():
                ref = []
                for ele in gt_dict[key]:
                    ref.append(ele['caption'].split())
                    '''print("GT: "+ele['caption'])'''
                references.append(ref)
                #print("PD:  " + pred_dict[key][0]['caption'])
                predictions.append(pred_dict[key][0]['caption'].split())

            references_test=[]
            predictions_test=[]
            for key in pred_dict_test.keys():
                ref = []
                for ele in gt_dict_test[key]:
                    ref.append(ele['caption'].split())
                    print("GT: "+ele['caption'])
                references_test.append(ref)
                print("PD:  " + pred_dict_test[key][0]['caption'])
                predictions_test.append(pred_dict_test[key][0]['caption'].split())

            score = corpus_bleu(references, predictions, weights=(0,0,0,1))
            print("Bleu4: %f"%score)
            score = corpus_bleu(references, predictions, weights=(0,0,1,0))
            print("Bleu3: %f"%score)
            score = corpus_bleu(references, predictions, weights=(0.25,0.25,0.25,0.25))
            print("Bleu: %f"%score)
            score = corpus_bleu(references_test, predictions_test, weights=(0,0,0,1))
            print("Bleu4test: %f"%score)
            score = corpus_bleu(references_test, predictions_test, weights=(0,0,1,0))
            print("Bleu3: %f"%score)
            score = corpus_bleu(references_test, predictions_test, weights=(0.25,0.25,0.25,0.25))
            print("Bleu: %f"%score)
            scorer = COCOScorer()
            total_score = scorer.score(gt_dict, pred_dict, id_list)
            scorer = COCOScorer()
            total_score = scorer.score(gt_dict_test, pred_dict_test, id_list_test)
        sys.stdout.flush()
    #print("Finally, saving the model ...")
    #saver.save(sess, os.path.join(model_path, 'model'), global_step=n_epochs)
    print("done")

train()

