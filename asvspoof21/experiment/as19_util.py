########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging
import math

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from librosa import util

key_encoder = {'bonafide': 0, 'spoof': 1}
la_system_encoder = {'-': 0, 'A01': 1, 'A02': 2, 'A03': 3, 'A04': 4, 'A05': 5, 'A06': 6, 'A07': 7, 'A08': 8, 'A09': 9,
                     'A10': 10, 'A11': 11, 'A12': 12, 'A13': 13, 'A14': 14, 'A15': 15, 'A16': 16, 'A17': 17,
                     'A18': 18, 'A19': 19}
pa_env_encoder = {'aaa': 0, 'aab': 1, 'aac': 2, 'aba': 3, 'abb': 4, 'abc': 5, 'aca': 6, 'acb': 7, 'acc': 8,
                  'baa': 9, 'bab': 10, 'bac': 11, 'bba': 12, 'bbb': 13, 'bbc': 14, 'bca': 15, 'bcb': 16, 'bcc': 17,
                  'caa': 18, 'cab': 19, 'cac': 20, 'cba': 21, 'cbb': 22, 'cbc': 23, 'cca': 24, 'ccb': 25, 'ccc': 26}
pa_attack_encoder = {'-': 0, 'AA': 1, 'AB': 2, 'AC': 3, 'BA': 4, 'BB': 5, 'BC': 6, 'CA': 7, 'CB': 8, 'CC': 9}


def get_label_id_one(label_type, env, attack, key):
    return [0.0 if label == 'spoof' else 1.0 for label in key]


def get_label_id_two(label_type, env, attack, key):
    label_type = label_type.lower()

    if label_type == 'key':
        label_id = [key_encoder.get(label) for label in key]
    elif label_type == 'env':
        label_id = [pa_env_encoder.get(label) for label in env]
    elif label_type == 'attack':
        label_id = [pa_attack_encoder.get(label) for label in attack]
    elif label_type == 'system':
        label_id = [la_system_encoder.get(label) for label in attack]
    else:
        raise Exception()

    return np.array(label_id)


def get_mt_label_id(env, attack, key, access_type):
    key_label_id = [key_encoder.get(label) for label in key]

    if access_type == 'LA':
        attack_label_id = [la_system_encoder.get(label) for label in attack]
    elif access_type == 'PA':
        attack_label_id = [pa_attack_encoder.get(label) for label in attack]
    else:
        raise ValueError()

    # env_label_id = [pa_env_encoder.get(label) for label in env]

    # label_id = [[0] for i in range(len(key_label_id))]
    label_id = np.ndarray((len(key_label_id), 2), dtype=np.long)
    for idx in range(len(key_label_id)):
        label_id[idx, 0] = key_label_id[idx]
        label_id[idx, 1] = attack_label_id[idx]  # , env_label_id[idx]

    return label_id


def get_num_classes(label_type):
    label_type = label_type.lower()

    if label_type == 'key':
        return 2
    elif label_type == 'env':
        return 27
    elif label_type == 'attack':
        return 10
    elif label_type == 'system':
        return 7
    else:
        raise Exception()


def read_protocol(protocol_file):
    # SPEAKER_ID  AUDIO_FILE_NAME ENVIRONMENT_ID ATTACK_ID(SYSTEM_ID)  KEY
    protocol = pd.read_csv(protocol_file, header=None, sep=r' ', dtype=str, engine='python',
                           names=['speaker_id', 'audio_file_name', 'env_id', 'attack_id', 'key'])
    speaker_id = protocol['speaker_id']
    audio_file_name = protocol['audio_file_name']
    env_id = protocol['env_id']
    attack_id = protocol['attack_id']
    key = protocol['key']

    return speaker_id.values, audio_file_name.values, env_id.values, attack_id.values, key.values


def read_eval_protocol(protocol_file):
    protocol = pd.read_csv(protocol_file, header=None, sep=r' ', dtype=str, engine='python',
                           names=['audio_file_name'])
    audio_file_name = protocol['audio_file_name']

    return audio_file_name.values


def save_scores4(score_file_name, utterance, system_id, key, score):
    with open(score_file_name, 'wt') as f:
        for i in range(len(utterance)):
            f.write('%s %s %s %.6f\n' % (utterance[i], system_id[i], key[i], score[i]))


def save_scores(file_name, segment_ids, scores):
    with open(file_name, 'wt') as f:
        for i in range(len(segment_ids)):
            f.write('%s %.6f\n' % (segment_ids[i], scores[i]))


def show_label_count(label_id):
    label_unique = np.unique(label_id)
    for label_name in label_unique:
        logging.info('{}:{}'.format(label_name, len(np.where(label_id == label_name)[0])))
    logging.info('All:{}'.format(len(label_id)))


def display_statistics(mylist):
    result = {}
    for item in set(mylist):
        result[item] = mylist.count(item)
    logging.info(result)


def frame_pt(x, frame_length, hop_length, axis=-1):
    # if not isinstance(x, np.ndarray):
    #     raise ParameterError(
    #         "Input must be of type numpy.ndarray, " "given type(x)={}".format(type(x))
    #     )
    #
    # if x.shape[axis] < frame_length:
    #     raise ParameterError(
    #         "Input is too short (n={:d})"
    #         " for frame_length={:d}".format(x.shape[axis], frame_length)
    #     )
    #
    # if hop_length < 1:
    #     raise ParameterError("Invalid hop_length: {:d}".format(hop_length))
    #
    # if axis == -1 and not x.flags["F_CONTIGUOUS"]:
    #     warnings.warn(
    #         "librosa.util.frame called with axis={} "
    #         "on a non-contiguous input. This will result in a copy.".format(axis)
    #     )
    #     x = np.asfortranarray(x)
    # elif axis == 0 and not x.flags["C_CONTIGUOUS"]:
    #     warnings.warn(
    #         "librosa.util.frame called with axis={} "
    #         "on a non-contiguous input. This will result in a copy.".format(axis)
    #     )
    #     x = np.ascontiguousarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    # strides = np.asarray(x.stride())
    xstrides = x.stride()
    # strides = torch.tensor(x.stride)

    # itemsize = x.
    # new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize
    # new_stride = np.prod(strides)

    if axis == -1:
        shape = [n_frames, ] + list(x.shape)[:-1] + [frame_length]
        strides = [hop_length] + list(xstrides)
        # shape = list(x.shape)[:-1] + [frame_length, n_frames]
        # strides = [hop_length * new_stride] + list(strides)

    # elif axis == 0:
    #     shape = [n_frames, frame_length] + list(x.shape)[1:]
    #     strides = [hop_length * new_stride] + list(strides)

    # else:
    #     raise ParameterError("Frame axis={} must be either 0 or -1".format(axis))

    return torch.as_strided(x, size=shape, stride=strides)


def segment_feature(data, frame_length, hop_length):
    data_len = data.shape[-1]
    if data_len < frame_length:
        # data = data.repeat(1, math.ceil(frame_length / data_len))
        # data = np.tile(data, (1, math.ceil(frame_length / data.shape[1])))
        if data.ndim == 2:
            data = data.repeat(1, math.ceil(frame_length / data_len))
            data = data[:, 0:frame_length]
        elif data.ndim == 3:
            data = data.repeat(1, 1, math.ceil(frame_length / data_len))
            data = data[:, :, 0:frame_length]
    else:
        num = math.ceil(data_len / hop_length) * hop_length - data_len
        if data.ndim == 2:
            data = torch.cat((data, data[:, :num]), dim=1)
            # data = torch.hstack((data, data[:, :num]))
        elif data.ndim == 3:
            data = torch.cat((data, data[:, :, :num]), dim=2)
            # data = torch.hstack((data, data[:, :num]))

    data = frame_pt(data, frame_length=frame_length, hop_length=hop_length, axis=-1)
    return data


def segment_np(data, frame_length, hop_length):
    data_len = data.shape[-1]
    if data_len < frame_length:
        if data.ndim == 2:
            data = np.tile(data, (1, math.ceil(frame_length / data_len)))
            data = data[:, 0:frame_length]
        elif data.ndim == 3:
            data = np.tile(data, (1, 1, math.ceil(frame_length / data_len)))
            data = data[:, :, 0:frame_length]
    else:
        num = math.ceil(data_len / hop_length) * hop_length - data_len
        if data.ndim == 2:
            data = np.concatenate((data, data[:, :num]), axis=1)
        elif data.ndim == 3:
            data = np.concatenate((data, data[:, :, :num]), axis=2)

    data = util.frame(data, frame_length=frame_length, hop_length=hop_length, axis=-1)
    return np.transpose(data, (2, 0, 1))


def segment_count(data, frame_length, hop_length):
    if data.shape[1] < frame_length:
        return 1
    else:
        return math.ceil(data.shape[1] / hop_length) - 1


def ufm_collate_fn(batch):
    datalist, labels = [], []

    for data, label in batch:
        datalist.append(data)
        labels.append(label)

    return datalist, labels


if __name__ == '__main__':
    # train_file, train_type = read_protocol(r'Z:\experiment\DS_10283_3055\protocol_V2\ASVspoof2017_V2_train.trn.txt')
    # train_feat = load_feat_list(r'D:\LZC\experiment\feat\spoof2017_CQCC_train.h5', train_file)

    filename = '/home/lzc/lzc/ASVspoof2019/ASVspoof2019feat/LFCC/LFCC_LA_train/LA_T_1000137.h5'
    with h5py.File(filename, 'r') as f:
        data = f['data'][:]
    data = torch.from_numpy(data).float()

    # aa = data * data

    aa = segment_feature(data, 11, 7)

    aa = 1
