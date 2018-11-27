import numpy as np
import os, json, h5py, math, glob
import tensorflow as tf
from tensorcv.models.layers import conv, max_pool, global_avg_pool
import cv2
from tensorflow.contrib.framework import add_arg_scope

MAX_LEN = 28
LSTM_DIM = 512
BATCH_SIZE = 64
inp_path = 'msvd_data'
h5py_path = 'h5py'
label_path = 'labels_complete'
splitdataset_path = 'msvd_dataset.npz'
chunk = 'ch10'
video_path='YouTubeClips'
pathSave='Images'                                                   #image frames

@add_arg_scope
def inception_layer(inputs,
                    conv_11_size,
                    conv_33_reduce_size, conv_33_size,
                    conv_55_reduce_size, conv_55_size,
                    pool_size,
                    data_dict={},
                    trainable=False,
                    name='inception'):
    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([conv], nl=tf.nn.relu, trainable=trainable,
                   data_dict=data_dict):
        conv_11 = conv(inputs, 1, conv_11_size, '{}_1x1'.format(name))

        conv_33_reduce = conv(inputs, 1, conv_33_reduce_size,
                              '{}_3x3_reduce'.format(name))
        conv_33 = conv(conv_33_reduce, 3, conv_33_size, '{}_3x3'.format(name))

        conv_55_reduce = conv(inputs, 1, conv_55_reduce_size,
                              '{}_5x5_reduce'.format(name))
        conv_55 = conv(conv_55_reduce, 5, conv_55_size, '{}_5x5'.format(name))

        pool = max_pool(inputs, '{}_pool'.format(name), stride=1,
                        padding='SAME', filter_size=3)
        convpool = conv(pool, 1, pool_size, '{}_pool_proj'.format(name))

    return tf.concat([conv_11, conv_33, conv_55, convpool],
                     3, name='{}_concat'.format(name))


# In[3]:


def _create_conv(inputs, data_dict):
    arg_scope = tf.contrib.framework.arg_scope

    with arg_scope([conv], trainable=False, data_dict=data_dict, nl=tf.nn.relu):
        conv1 = conv(inputs, 7, 64, name='conv1_7x7_s2', stride=2)
        padding1 = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        conv1_pad = tf.pad(conv1, padding1, 'CONSTANT')
        pool1 = max_pool(conv1_pad, 'pool1', padding='VALID', filter_size=3, stride=2)
        pool1_lrn = tf.nn.local_response_normalization(pool1, depth_radius=2, alpha=2e-05, beta=0.75, name='pool1_lrn')

        conv2_reduce = conv(pool1_lrn, 1, 64, name='conv2_3x3_reduce')
        conv2 = conv(conv2_reduce, 3, 192, name='conv2_3x3')
        padding2 = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        conv2_pad = tf.pad(conv2, padding1, 'CONSTANT')
        pool2 = max_pool(conv2_pad, 'pool2', padding='VALID', filter_size=3, stride=2)
        pool2_lrn = tf.nn.local_response_normalization(pool2, depth_radius=2, alpha=2e-05, beta=0.75, name='pool2_lrn')
    with arg_scope([inception_layer], trainable=False, data_dict=data_dict):
        inception3a = inception_layer(
            pool2_lrn, 64, 96, 128, 16, 32, 32, name='inception_3a')
        inception3b = inception_layer(
            inception3a, 128, 128, 192, 32, 96, 64, name='inception_3b')
        pool3 = max_pool(
            inception3b, 'pool3', padding='SAME', filter_size=3, stride=2)

        inception4a = inception_layer(
            pool3, 192, 96, 208, 16, 48, 64, name='inception_4a')
        inception4b = inception_layer(
            inception4a, 160, 112, 224, 24, 64, 64, name='inception_4b')
        inception4c = inception_layer(
            inception4b, 128, 128, 256, 24, 64, 64, name='inception_4c')
        inception4d = inception_layer(
            inception4c, 112, 144, 288, 32, 64, 64, name='inception_4d')
        inception4e = inception_layer(
            inception4d, 256, 160, 320, 32, 128, 128, name='inception_4e')
        pool4 = max_pool(
            inception4e, 'pool4', padding='SAME', filter_size=3, stride=2)

        inception5a = inception_layer(
            pool4, 256, 160, 320, 32, 128, 128, name='inception_5a')
        inception5b = inception_layer(
            inception5a, 384, 192, 384, 48, 128, 128, name='inception_5b')
        pool5=global_avg_pool(inception5b,name='global_avg_pool')

    return pool5


# ## main function

# In[4]:


def init_trained_weight():
    import h5py
    filename = 'googlenet_weights.h5'
    f = h5py.File(filename, 'r')

    # List all groups
    #print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data_dict = list(f[a_group_key])

    return data_dict



# In[7]:


def google_net(List, i_path, load_weight=True):
    data_dict = {}
    if load_weight:
        data_dict = np.load('googlenet.npy',encoding='latin1').item()
    else:
        data_dict = {}

    MEAN = [103.939, 116.779, 123.68]
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=image)
    input_bgr = tf.concat(axis=3, values=[blue - MEAN[0], green - MEAN[1], red - MEAN[2], ])
    inception5b = _create_conv(input_bgr, data_dict)
    init = tf.global_variables_initializer()
    v_all=[]
    i=1
    with tf.Session() as sess:
        sess.run(init)
        for vdo in List:
            print(i)
            im_count = makeFrame(vdo)
            im = load_images_from_folder(i_path)
            v = sess.run(inception5b, {image: im})
            print(v.shape)
            v = np.reshape(v, [-1, 1024])
            v = get_appended_features(v, im_count)     #total_videos*28*
            #print(v.shape)
            v_all.append(v)
            os.system("rm Images/*")
            i=i+1
        
        #print(np.asarray(w).shape)

    return v_all
def get_max_len(path):
    lst = []
    for root, dirs, files in os.walk(path):
        for ele in files:
            if ele.endswith('json'):
                lst.append(root + '/' + ele)
    cnt = []
    for ele in lst:
        a = json.load(open(ele))
        cnt.append(len(a[0]))
    return max(cnt)

def makeFrame(file):
    im_count=0
    f_path = os.path.join(video_path, file+'.avi')
    if not os.path.exists(f_path):
        return []
    vidcap = cv2.VideoCapture(f_path)
    frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    np.random.seed(9001)
    #s = np.random.randint(1, frameCount + 1, 28)
    s = np.linspace(1, frameCount + 1, num=MAX_LEN+1, dtype=np.int)
    count = 0
    fno = 0
    count1 = 0
    success, image = vidcap.read()
    while success:
        count += 1
        if count in s:
            resized_image = cv2.resize(image, (224, 224))
            cv2.imwrite(os.path.join(pathSave, "frame%d.jpg" % (fno)), resized_image)
            count1 = count1 + 1
            fno = fno + 1
        success, image = vidcap.read()
    im_count+=count1
    return im_count

def load_images_from_folder(folder):
    ima = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            ima.append(img)
    return ima

def get_features(i_path):
    if not os.path.exists(i_path):
        return []
    v=[]
    im=load_images_from_folder(i_path)
    train_data = google_net(im, True)
    v.append(train_data)

    return v


def check_HL_nonHL_exist(label):
    idx = len(np.where(label == 1)[0])
    idy = len(np.where(label == 0)[0])
    return idx > 0 and idy > 0

def generate_h5py(X, fname, dataset, feature_folder_name, batch_start=0):
    X=np.transpose(X,[1,0,2])
    dirname = os.path.join(h5py_path, feature_folder_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    q_idx=0
    num = X.shape[1]
    if num % BATCH_SIZE == 0:
        batch_num = int(num / BATCH_SIZE)
    else:
        batch_num = int(num / BATCH_SIZE) + 1
    f_txt = open(os.path.join(dirname, dataset + '.txt'), 'w')
    mapping = json.load(open('msvd2sent.json'))
    for i in range(batch_start, batch_start + batch_num):
        train_filename = os.path.join(dirname, dataset + str(i) + '.h5')
        if os.path.isfile(train_filename):
            q_idx += BATCH_SIZE
            continue
        with h5py.File(train_filename, 'w') as f:
            f['data'] = np.zeros([MAX_LEN, BATCH_SIZE, X.shape[2]])         #28*b*50176
            f['cont'] = np.zeros([MAX_LEN, BATCH_SIZE])
            f['reindex'] = np.zeros(MAX_LEN)
            fname_tmp = []
            title_tmp = []
            for j in range(BATCH_SIZE):
                if q_idx >= X.shape[1]:
                    continue
                f['data'][:, j, :] = X[:,q_idx, :]
                f['cont'][1:MAX_LEN + 1, j] = 1
                f['reindex'][:MAX_LEN] = np.arange(MAX_LEN)
                f['reindex'][MAX_LEN:] = MAX_LEN
                fname_tmp.append(fname[q_idx])
                title_tmp.append(fname[q_idx])
                if q_idx == X.shape[1]:
                    while len(fname_tmp) < BATCH_SIZE:
                        fname_tmp.append('')
                        title_tmp.append('')
                    fname_tmp = np.array(fname_tmp)
                    title_tmp = np.array(title_tmp)
                    f['fname'] = fname_tmp
                    f['title'] = title_tmp
                    f_txt.write(train_filename + '\n')
                    return
                q_idx += 1
            fname_tmp = np.array(fname_tmp)
            title_tmp = np.array(title_tmp)
            #f['fname'] = fname_tmp
            f['fname']=[a.encode('utf8') for a in fname_tmp]
            f['title'] = [a.encode('utf8') for a in title_tmp]
            # f.create_dataset('title', data=title_tmp)
        f_txt.write(train_filename + '\n')




def generate_npz(X, q, fname, dataset, feature_folder_name):
    dirname = os.path.join(inp_path, 'data_' + chunk, 'npz', feature_folder_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = os.path.join(dirname, dataset)
    np.savez(filename, fv=X, q=q, fname=fname)


def get_feats_depend_on_label(label, per_f, v, idx):
    X = []
    y = []
    q = []
    for l_index in range(len(label[0])):
        low = int(math.ceil(label[0][l_index][0] / per_f))
        up = min(len(v), int(math.ceil(label[0][l_index][1] / per_f)))
        up_ = up
        # pdb.set_trace()
        if low >= len(v) or low == up:
            X.append(X[-1])
        else:
            ## take the mean feature of frames in one clip
            X.append(np.mean(v[low:up, :], axis=0))
            ## random sample feature of frames in one clip
        #            X.append(v[np.random.randint(low,up),:])

        y.append(label[1][l_index])
        q.append(idx)
    return X, y, q

def get_appended_features(v,im_count):
    #v=np.reshape(v,[np.shape(v)[1],-1])
    tempZeros = np.zeros((MAX_LEN - im_count, 1024))
    v=np.append(v,tempZeros,0)
    return v



def load_feats(vTemp, List):
    X = vTemp
    q = []
    fname = []

    for ix in range(np.size(List)):
        q += MAX_LEN * [ix]
        fname.append(List[ix])
    # pdb.set_trace()
    return np.array(X), np.array(q), np.array(fname)


def Normalize(X, normal=0):
    if normal == 0:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        idx = np.where(std == 0)[0]
        std[idx] = 1
    else:
        mean = normal[0]
        std = normal[1]
    X = (X - mean) / std
    return X, mean, std


def driver(inp_type, outp_folder_name):
    List = np.append(np.append(np.load(splitdataset_path)['train'], np.load(splitdataset_path)['val']),
                     np.load(splitdataset_path)['test'])
    v_all = google_net(List, pathSave, True)
    v_all=np.reshape(v_all,[-1,MAX_LEN,1024])
    print(v_all.shape)
    '''v_all=[]
    for vdo in List:
        im_count = makeFrame(vdo)
        v = get_features(pathSave)
        v = get_appended_features(v, im_count)     #total_videos*28*1024
        v_all.append(v)
        os.system("rm Images/*")
        tf.get_variable_scope().reuse_variables()'''
    dataset = 'train'
    # List = open(os.path.join('data_' + chunk, dataset + '_list.txt'),'r').read().split('\n')[:-1]
    List1 = np.load(splitdataset_path)[dataset]
    v1=v_all[0:len(List1)]                    #[num_videos,28,1024]
    print(np.shape(v1))
    print(List1)
    for iii in range(int(math.ceil(len(List1) / 512.))):
        [X, Q, fname] = load_feats(v1[iii * 640:min(len(List1), (iii + 1) * 512)], List1[iii * 512:min(len(List1), (iii + 1) * 512)])
        if inp_type == 'h5py':
            generate_h5py(X, fname, dataset, outp_folder_name, batch_start=iii * 512)
        else:
            generate_npz(X, Q, fname, dataset, outp_folder_name)

    dataset = 'val'
    List2 = np.load(splitdataset_path)[dataset]
    v2 = v_all[len(List1):len(List1)+len(List2)]
    print(np.shape(v2))
    print(List2)
    for iii in range(int(math.ceil(len(List2) / 512.))):
        [X, Q, fname] = load_feats(v2[iii * 512:min(len(List2), (iii + 1) * 512)], List2[iii * 512:min(len(List2), (iii + 1) * 512)])
        # [X, mean, std] = Normalize(X, [mean, std])
        if inp_type == 'h5py':
            generate_h5py(X, fname, dataset, outp_folder_name, batch_start=iii * 512)
        else:
            generate_npz(X, Q, fname, dataset, outp_folder_name)

    dataset = 'test'
    List3 = np.load(splitdataset_path)[dataset]
    v3 = v_all[len(List1) + len(List2):len(List1) + len(List2)+len(List3)]
    print(np.shape(v3))
    print(List3)
    for iii in range(int(math.ceil(len(List3) / 512.))):
        [X, Q, fname] = load_feats(v3[iii * 512:min(len(List3), (iii + 1) * 512)],
                                   List3[iii * 512:min(len(List3), (iii + 1) * 512)])
        # [X, mean, std] = Normalize(X, [mean, std])
        if inp_type == 'h5py':
            generate_h5py(X, fname, dataset, outp_folder_name, batch_start=iii * 512)
        else:
            generate_npz(X, Q, fname, dataset, outp_folder_name)


def getlist(path, split):
    List = glob.glob(path + split + '*.h5')
    f = open(path + split + '.txt', 'w')
    for ele in List:
        f.write(ele + '\n')

image = tf.placeholder(tf.float32, shape=[None, None, None, 3])

if __name__ == '__main__':
    driver('h5py',  'cont')
    path = os.path.join(h5py_path, 'cont' + '/')
    getlist(path, 'train')
    getlist(path, 'val')
    getlist(path, 'test')

