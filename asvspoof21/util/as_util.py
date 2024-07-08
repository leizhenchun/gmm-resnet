########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging
import os
import shutil
import sys
import zipfile

import h5py
import numpy
import torch


def is_debug():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    elif gettrace():
        return True
    else:
        return False


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def random_seed(seed=0):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dir(dir_name):
    if not os.path.isdir(dir_name):
        # os.mkdir(path_name)
        os.makedirs(dir_name)


def clear_dir(dir_name):
    logging.info('Clear dir : ' + dir_name)
    shutil.rmtree(dir_name, ignore_errors=True)
    os.makedirs(dir_name)


def load_array_h5(filename, data_name='data'):
    with h5py.File(filename, 'r') as f:
        data = f[data_name][:]
    return data


def save_array_h5(filename, data, data_name='data'):
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as f:
        f[data_name] = data


def zip_score_file(score_file, compresslevel=9):
    file_path, file_name = os.path.split(score_file)
    short_name, extension = os.path.splitext(file_name)
    # file_name, file_extension = os.path.splitext(score_file)

    try:
        with zipfile.ZipFile(file_path + '/' + short_name + '.zip', mode="w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=compresslevel) as f:
            f.write(score_file, arcname=short_name + '.txt')
    except Exception as e:
        logging.info(e)
    finally:
        f.close()


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def get_parameter_number_details(net):
    trainable_num_details = {name: p.numel() for name, p in net.named_parameters() if p.requires_grad}
    return trainable_num_details


def show_model(model):
    logging.info(model)
    # model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logging.info('Model total parameter: {:,}'.format(model_params))

    # logging.info(get_parameter_number_details(model))
    total_num, trainable_num = get_parameter_number(model)
    logging.info('Total params: {:,}      Trainable params: {:,}'.format(total_num, trainable_num))


def args2str(args):
    result = []
    for arg in vars(args):
        result.append('{} : {}\n'.format(arg, getattr(args, arg)))
    return ''.join(result)


if __name__ == '__main__':
    # train_file, train_type = read_protocol(r'Z:\experiment\DS_10283_3055\protocol_V2\ASVspoof2017_V2_train.trn.txt')
    # train_feat = load_feat_list(r'D:\LZC\experiment\feat\spoof2017_CQCC_train.h5', train_file)

    filename = '/home/lzc/lzc/ASVspoof2019/ASVspoof2019feat2/LFCC/LFCC_LA_train/LA_T_1000137.h5'
    zip_score_file(filename)
