import os
import cv2
import json
#import pdb
import numpy as np

video_path = 'YouTubeClips'
output_path = 'labels_complete/'

train_size=1200
val_size=100
# Split data to train data, valid data and test data
def splitdata(path, train_num, val_num):
    lst = os.listdir(path)
    name = []
    for ele in lst:
        name.append(os.path.splitext(ele)[0])

    print(name)
    train = name[0:train_num]
    val = name[train_num:train_num + val_num]
    test = name[train_num + val_num:]
    np.savez('msvd_dataset', train=train, val=val, test=test)


def get_total_frame_number(fn):
    cap = cv2.VideoCapture(fn)
    if not cap.isOpened():
        print
        "could not open :", fn
        sys.exit()
    length = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


def getlist(Dir):
    pylist = []
    for root, dirs, files in os.walk(Dir):
        for ele in files:
            if ele.endswith('avi'):
                pylist.append(root + '/' + ele)

    return pylist


def get_frame_list(frame_num):
    start = 0.0
    i = 0
    end = 0.0
    frame_list = []
    if frame_num > 450:
        frame_num = 450.0
    while end < frame_num:
        start = 10.0 * i
        end = start + 10.0
        i += 1
        if end > frame_num:
            end = frame_num
        frame_list.append([start, end])
    return frame_list


if __name__ == '__main__':
    splitdata(video_path, train_size, val_size )
    b = getlist(video_path)


