########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging
import os
import shutil
from functools import partial
from multiprocessing import Pool

import h5py
import numpy
import torch
import torchaudio
import torchaudio.backend.soundfile_backend
from tqdm import tqdm

from augmentation.as_data_augmentation import augment_audio
from feature.as_feature import extract_feature
from asvspoof19.as19_util import read_protocol
from util.util import clear_dir

if os.name == 'posix':
    data_path_19 = '/home/labuser/ssd/lzc/ASVspoof/DS_10283_3336'
    feat_path_19 = '/home/labuser/ssd/lzc/ASVspoof/ASVspoof2019feat'
    data_path_21 = '/home/labuser/ssd/lzc/ASVspoof/ASVspoof2021data'
    feat_path_21 = '/home/labuser/ssd/lzc/ASVspoof/ASVspoof2021feat'
elif os.name == 'nt':
    data_path_19 = '/home/lzc/lzc/ASVspoof/DS_10283_3336'
    feat_path_19 = '/home/lzc/lzc/ASVspoof/ASVspoof2019feat'
    data_path_21 = '/home/lzc/lzc/ASVspoof/ASVspoof2021data'
    feat_path_21 = '/home/lzc/lzc/ASVspoof/ASVspoof2021feat'


def extract_one(utterance, wave_path, feat_path, feature_type, access_type, dataset, aug_methods):
    # wave_path = r'/home/lzc/lzc/ASVspoof2021/ASVspoof2019trainwave/LA/RB6'
    wave_file = os.path.join(wave_path, utterance + '.flac')
    audio, sr = torchaudio.load(wave_file)  # , normalize=True
    # audio = audio.double()

    for aug_method in aug_methods:
        # augmented_audio = augmented_audio.squeeze().detach().numpy()
        # augmented_audio = augmented_audio.astype(numpy.float32)

        augmented_audio = augment_audio(audio, sr, aug_method)
        # augmented_audio = augmented_audio.float()
        feature = extract_feature(augmented_audio, sr, feature_type, access_type)

        if dataset == 'train':
            feat_file = os.path.join(feat_path, aug_method, utterance + '.h5')
        else:
            feat_file = os.path.join(feat_path, utterance + '.h5')

        with h5py.File(feat_file, 'w') as f:
            feature = feature.squeeze(0).detach().numpy().T
            feature = feature.astype(numpy.float32)
            f['/data'] = feature


def extract_one_wav2vec(utterance, wave_path, feat_path, access_type, aug_methods, model=None, bundle_sample_rate=None):
    # wave_path = r'/home/lzc/lzc/ASVspoof2021/ASVspoof2019trainwave/LA/RB6'

    feature_type = 'WAV2VEC'

    wave_file = os.path.join(wave_path, utterance + '.flac')
    audio, sr = torchaudio.load(wave_file)
    # audio = audio.double()

    for aug_method in aug_methods:
        # augmented_audio = augment_audio(audio, sr, aug_method).squeeze().detach().numpy()
        # feature = extract_feature(augmented_audio, sr, feature_type, access_type)
        augmented_audio = augment_audio(audio, sr, aug_method)
        waveform = augmented_audio.cuda()  # to(device)

        if 16000 != bundle_sample_rate:
            waveform = torchaudio.functional.resample(waveform, 16000, bundle_sample_rate)

        with torch.inference_mode():
            features, _ = model.extract_features(waveform)

        feature = features[-1]
        feature = feature.cpu().squeeze().detach().numpy().T
        with h5py.File(os.path.join(feat_path, aug_method, utterance + '.h5'), 'w') as f:
            feature = feature.astype(numpy.float32)
            f['/data'] = feature


def extract_feature_list(utterances, wave_path, feat_path, feature_type, access_type, dataset, aug_methods=[]):
    logging.info('Source Path : {}'.format(wave_path))
    logging.info('Feature Path : {}'.format(feat_path))
    logging.info('Utterance Number : {}'.format(len(utterances)))

    if dataset == 'train':
        for aug_method in aug_methods:
            clear_dir(os.path.join(feat_path, aug_method))
    else:
        clear_dir(feat_path)

    extract_one_fun = partial(extract_one, wave_path=wave_path, feat_path=feat_path, feature_type=feature_type,
                              access_type=access_type, dataset=dataset, aug_methods=aug_methods)

    for utterance in tqdm(utterances):
        extract_one_fun(utterance)

    # with Pool(os.cpu_count()) as pool:
    #     list(tqdm(pool.imap(extract_one_fun, utterances)))

    logging.info('Done.')


def extract_wav2vec_list(utterances, wave_path, feat_path, access_type, aug_methods=[]):
    feature_type = 'WAV2VEC'
    logging.info('Feature Path : {}'.format(feat_path))
    logging.info('Utterance Number : {}'.format(len(utterances)))

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().cuda()  # .to(device)

    for aug_method in aug_methods:
        clear_dir(os.path.join(feat_path, aug_method))

    extract_one_fun = partial(extract_one_wav2vec, wave_path=wave_path, feat_path=feat_path,
                              access_type=access_type, aug_methods=aug_methods, model=model,
                              bundle_sample_rate=bundle.sample_rate)

    for utterance in tqdm(utterances):
        extract_one_fun(utterance)

    # with Pool(os.cpu_count()) as pool:
    #     list(tqdm(pool.imap(extract_one_fun, utterances)))

    logging.info('Done.')


def extract_feature_for_dataset(feature_type, access_type, dataset, data_path, feat_path, aug_methods=None):
    if dataset == 'train':
        protocol_file = os.path.join(data_path, access_type,
                                     'ASVspoof2019_{}_cm_protocols'.format(access_type),
                                     'ASVspoof2019.{}.cm.{}.trn.txt'.format(access_type, dataset))
    else:
        protocol_file = os.path.join(data_path, access_type,
                                     'ASVspoof2019_{}_cm_protocols'.format(access_type),
                                     'ASVspoof2019.{}.cm.{}.trl.txt'.format(access_type, dataset))
    spk, utterances, env, attack, key = read_protocol(protocol_file)

    wave_path = os.path.join(data_path, access_type, 'ASVspoof2019_{}_{}'.format(access_type, dataset), 'flac')
    feat_path = os.path.join(feat_path, feature_type, '{}_{}_{}'.format(feature_type, access_type, dataset))

    # / home / lzc / lzc / ASVspoof / ASVspoof2019feat / WAVE_VAD / WAVE_VAD_LA_train

    if feature_type.endswith('_VAD'):
        wave_path = os.path.join(r'/home/lzc/lzc/ASVspoof/ASVspoof2019feat/WAVE_VAD',
                                 'WAVE_VAD_{}_{}'.format(access_type, dataset))

        if dataset == 'train':
            wave_path = os.path.join(wave_path, 'Original')

    extract_feature_list(utterances, wave_path, feat_path, feature_type, access_type, dataset, aug_methods)


def extract_wav2vec_for_dataset(access_type, dataset, data_path, feat_path, aug_methods=None):
    if dataset == 'train':
        protocol_file = os.path.join(data_path, access_type,
                                     'ASVspoof2019_{}_cm_protocols'.format(access_type),
                                     'ASVspoof2019.{}.cm.{}.trn.txt'.format(access_type, dataset))
    else:
        protocol_file = os.path.join(data_path, access_type,
                                     'ASVspoof2019_{}_cm_protocols'.format(access_type),
                                     'ASVspoof2019.{}.cm.{}.trl.txt'.format(access_type, dataset))
    spk, utterances, env, attack, key = read_protocol(protocol_file)

    feature_type = 'WAV2VEC'

    wave_path = os.path.join(data_path, access_type, 'ASVspoof2019_{}_{}'.format(access_type, dataset), 'flac')
    feat_path = os.path.join(feat_path, feature_type, '{}_{}_{}'.format(feature_type, access_type, dataset))

    extract_wav2vec_list(utterances, wave_path, feat_path, access_type, aug_methods)


def extract_feature_asvspoof19(feature_type, access_type, aug_methods=None):
    extract_feature_for_dataset(feature_type, access_type, 'train', data_path_19, feat_path_19, aug_methods)

    if 'Original' in aug_methods:
        extract_feature_for_dataset(feature_type, access_type, 'dev', data_path_19, feat_path_19, ['Original'])
        extract_feature_for_dataset(feature_type, access_type, 'eval', data_path_19, feat_path_19, ['Original'])


def extract_wav2vec_asvspoof19(access_type, aug_methods=None):
    extract_wav2vec_for_dataset(access_type, 'train', data_path_19, feat_path_19, aug_methods)
    extract_wav2vec_for_dataset(access_type, 'dev', data_path_19, feat_path_19, ['Original'])
    extract_wav2vec_for_dataset(access_type, 'eval', data_path_19, feat_path_19, ['Original'])


def extract_feature_asvspoof21(feature_type, access_type):
    protocol_file = os.path.join(data_path_21, 'ASVspoof2021_{}_eval'.format(access_type),
                                 'ASVspoof2021.{}.cm.eval.trl.txt'.format(access_type))
    wave_path = os.path.join(data_path_21, 'ASVspoof2021_{}_eval'.format(access_type), 'flac')
    feat_path = os.path.join(feat_path_21, feature_type, '{}_{}_eval'.format(feature_type, access_type))

    if feature_type.endswith('_VAD'):
        wave_path = os.path.join(r'/home/lzc/lzc/ASVspoof/ASVspoof2021feat/WAVE_VAD',
                                 'WAVE_VAD_{}_eval'.format(access_type))

    utterances, _, _, _, _ = read_protocol(protocol_file)
    extract_feature_list(utterances, wave_path, feat_path, feature_type, access_type, 'eval', ['Original'])


def extract_wav2vec_asvspoof21(access_type):
    feature_type = 'WAV2VEC'
    protocol_file = os.path.join(data_path_21, 'ASVspoof2021_{}_eval'.format(access_type),
                                 'ASVspoof2021.{}.cm.eval.trl.txt'.format(access_type))
    wave_path = os.path.join(data_path_21, 'ASVspoof2021_{}_eval'.format(access_type), 'flac')
    feat_path = os.path.join(feat_path_21, feature_type, '{}_{}_eval'.format(feature_type, access_type))

    utterances, _, _, _, _ = read_protocol(protocol_file)
    extract_wav2vec_list(utterances, wave_path, feat_path, access_type, ['Original'])


def extract_wave(utterance, wave_path, wave_path_out, access_type, aug_methods):
    wave_file = os.path.join(wave_path, utterance + '.flac')
    # audio, sr = soundfile.read(wave_file)
    # audio = torch.from_numpy(audio).unsqueeze(0)
    audio, sr = torchaudio.load(wave_file)
    audio = audio.double()

    for aug_method in aug_methods:
        augmented_audio = augment_audio(audio, sr, aug_method)

        output_file = os.path.join(wave_path_out, access_type, aug_method, utterance + '.flac')
        torchaudio.save(filepath=output_file, src=augmented_audio, sample_rate=sr, format='flac')


def extract_wave_asvspoof19(data_path, wave_path_out, access_type, aug_methods=[]):
    protocol_file = os.path.join(data_path, access_type,
                                 'ASVspoof2019_{}_cm_protocols'.format(access_type),
                                 'ASVspoof2019.{}.cm.{}.trn.txt'.format(access_type, 'train'))

    spk, utterances, env, attack, key = read_protocol(protocol_file)

    wave_path = os.path.join(data_path, access_type, 'ASVspoof2019_{}_{}'.format(access_type, 'train'), 'flac')

    logging.info('Feature Path : {}'.format(wave_path_out))
    logging.info('Utterance Number : {}'.format(len(utterances)))

    for aug_method in aug_methods:
        clear_dir(os.path.join(wave_path_out, access_type, aug_method))

    extract_wave_fun = partial(extract_wave, wave_path=wave_path, wave_path_out=wave_path_out,
                               access_type=access_type, aug_methods=aug_methods)

    # for utterance in tqdm(utterances):
    #     extract_wave_fun(utterance)

    with Pool(os.cpu_count()) as pool:
        list(tqdm(pool.imap(extract_wave_fun, utterances)))

    logging.info('Done.')


# def extract_wave_asvspoof19(access_type, aug_methods=None):
#     extract_feature_for_dataset(feature_type, access_type, 'train', data_path_19, feat_path_19, aug_methods)
#
#     if 'Original' in aug_methods:
#         extract_feature_for_dataset(feature_type, access_type, 'dev', data_path_19, feat_path_19, ['Original'])
#         extract_feature_for_dataset(feature_type, access_type, 'eval', data_path_19, feat_path_19, ['Original'])


def rm_feat_dir(feat_path, feature_type, access_type):
    if 'ASVspoof2019feat' in feat_path:
        for subset in ['train', 'dev', 'eval']:
            feat_path1 = os.path.join(feat_path, feature_type, '{}_{}_{}'.format(feature_type, access_type, subset))
            logging.info('Delete dir : ' + feat_path1)
            shutil.rmtree(feat_path1, ignore_errors=True)
            # os.makedirs(feat_path)

    if 'ASVspoof2021feat' in feat_path:
        feat_path1 = os.path.join(feat_path, feature_type, '{}_{}_eval'.format(feature_type, access_type))
        logging.info('Delete dir : ' + feat_path1)
        shutil.rmtree(feat_path1, ignore_errors=True)
        # os.makedirs(feat_path)


def rm_feat_root(feat_path, feature_type):
    feat_path1 = os.path.join(feat_path, feature_type)
    logging.info('Delete dir : ' + feat_path1)
    shutil.rmtree(feat_path1, ignore_errors=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.NOTSET, format="[%(asctime)s - %(levelname)1.1s] %(message)s")

    # aug_methods = ['Original', 'ALAW', 'ULAW', 'MP3', 'VORBIS']
    # aug_methods = ['RB1', 'RB2', 'RB3', 'RB4', 'RB5', 'RB6', 'RB7', 'RB8']
    # aug_methods = ['RIR', 'NOISE', 'MUSIC', 'SPEECH', 'BABBLE']

    # aug_methods = ['Original', 'ALAW', 'ULAW', 'RB1', 'RB2', 'RB3', 'RB4', 'RB5', 'RB6', 'RB7', 'RB8', 'RIR', 'NOISE',
    #                'MUSIC', 'SPEECH']

    # aug_methods = ['Original', 'RB4']  # 'Original',RB4
    # feature_type = 'LFCC21NN_VAD'  # LFCC21NN

    # rm_feat_root(feat_path_19, feature_type)
    # rm_feat_root(feat_path_21, feature_type)

    # rm_feat_dir(feat_path_19, feature_type, 'PA')
    # rm_feat_dir(feat_path_21, feature_type, 'PA')

    feature_type = 'LFCC19'  # LFCC21NN_VAD'  # LFCC21NN
    aug_methods = ['Original', 'RB4']  # 'Original', 'RB4'
    extract_feature_asvspoof19(feature_type, access_type='LA', aug_methods=aug_methods)
    # extract_feature_asvspoof21(feature_type, access_type='LA')
    # extract_feature_asvspoof21(feature_type, access_type='DF')

    extract_feature_asvspoof19(feature_type, access_type='PA', aug_methods=['Original', ])
    extract_feature_asvspoof21(feature_type, access_type='PA')

    # aug_methods = ['Original']  # 'Original',RB4
    # extract_feature_asvspoof21(feature_type, access_type='PA')

    # wave_path_out = r'/home/lzc/lzc/ASVspoof2021/ASVspoof2019trainwave'
    # extract_wave_asvspoof19(data_path_19, wave_path_out, 'LA', aug_methods=aug_methods)

    # aug_methods = ['Original', ]
    # extract_wav2vec_asvspoof19(access_type='LA', aug_methods=aug_methods)
    # extract_wav2vec_asvspoof21(access_type='LA')
    # extract_wav2vec_asvspoof21(access_type='DF')
