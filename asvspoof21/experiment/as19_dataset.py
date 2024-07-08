########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import glob
import logging
import math
import os
import random

import h5py
# import librosa
import numpy
import numpy as np
import soundfile
import torch
import torchaudio
from scipy import signal
from torch.utils.data import Dataset

from asvspoof21.augmentation.as_data_augmentation import augment_audio
from asvspoof21.feature.as_feature import extract_feature


# def trim_feat(data, frame_length):
#     if data.shape[1] < frame_length:
#         data = np.tile(data, (1, math.ceil(frame_length / data.shape[1])))
#     return data[:, 0:frame_length]


def cut_wave(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def feature_trim_pad(data, frame_length, random_select=False):
    if data.shape[-1] < frame_length:
        if data.ndim == 1:
            data = np.tile(data, (math.ceil(frame_length / data.shape[0])))
            # data = data[0:frame_length]
        elif data.ndim == 2:
            data = np.tile(data, (1, math.ceil(frame_length / data.shape[1])))
            # data = data[:, 0:frame_length]
        elif data.ndim == 3:
            data = np.tile(data, (1, 1, math.ceil(frame_length / data.shape[1])))
            # data = data[:, :, 0:frame_length]

        # data = np.pad(data, pad_width=((0, 0), (0, frame_length - data.shape[1])), mode='constant', constant_values=0.0)

    feat_len = data.shape[-1]
    if random_select and feat_len > frame_length:
        start = np.random.randint(feat_len - frame_length)
    else:
        start = 0

    # if data.shape[-1] > frame_length:
    if data.ndim == 1:
        data = data[start:start + frame_length]
    elif data.ndim == 2:
        data = data[:, start:start + frame_length]
    elif data.ndim == 3:
        data = data[:, :, start:start + frame_length]

    return data


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename, dtype='float32')

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = np.stack(feats, axis=0)  # .astype(numpy.float)

    return feat


def segmentWAV(audio, max_audio, evalmode=True, num_eval=10):
    # Maximum audio length
    # max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    # audio, sample_rate = soundfile.read(filename, dtype='float32')

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and max_audio == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = np.stack(feats, axis=0)  # .astype(numpy.float)

    return feat


class AugmentWAV2(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise', 'speech', 'music']

        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))

        logging.info('Loading musan wave ....')
        self.musan_waves = {}
        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
                self.musan_waves[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

            noise, fs = soundfile.read(file, dtype='float32')
            self.musan_waves[file.split('/')[-4]].append(noise)
        logging.info('Done!   len : {}'.format(len(augment_files)))

        logging.info('Loading RIR wave ....')
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
        self.rir_waves = []
        for rir_file in self.rir_files:
            rir, fs = soundfile.read(rir_file, dtype='float32')
            rir = np.expand_dims(rir, 0)
            rir = rir / np.sqrt(np.sum(rir ** 2))
            self.rir_waves.append(rir)
        logging.info('Done!   len : {}'.format(len(self.rir_files)))

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        # noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noiselist = random.sample(self.musan_waves[noisecat], random.randint(numnoise[0], numnoise[1]))

        noises = []

        for noise in noiselist:
            # noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noiseaudio = segmentWAV(noise, audio.shape[1], evalmode=False)

            # noiseaudio = cut_wave(noise, self.max_audio)

            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        noise = np.concatenate(noises, axis=0)
        noise = np.sum(noise, axis=0, keepdims=True)
        return noise + audio

    def reverberate(self, audio):
        # rir_file = random.choice(self.rir_files)
        # rir, fs = soundfile.read(rir_file, dtype='float32')
        # rir = numpy.expand_dims(rir, 0)
        # rir = rir / numpy.sqrt(numpy.sum(rir ** 2))

        rir = random.choice(self.rir_waves)

        return signal.convolve(audio, rir, mode='full')[:, :self.max_audio]


class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio = max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise', 'speech', 'music']

        # self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        # self.numnoise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}
        self.noisesnr = {'noise': [5, 20], 'speech': [5, 20], 'music': [5, 20]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}

        self.noiselist = {}

        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))

        noises = []

        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):

        rir_file = random.choice(self.rir_files)

        rir, fs = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))

        return signal.convolve(audio, rir, mode='full')[:, :self.max_audio]


class ASVSpoofDataset(Dataset):
    def __init__(self, feature_file_path, feature_file_list, feature_file_extension='.h5', label_id=None, feat_length=0,
                 num_channels=0, transpose=False, keep_in_memory=False, sample_ratio=1.0, feat_transformer_fn=None,
                 normalization=False,
                 dtype=np.float32):
        super(ASVSpoofDataset, self).__init__()

        self.dtype = dtype
        self.sample_ratio = sample_ratio
        # if 0 < sample_ratio < 1.0:
        #     feature_file_list, label_id = random_select_sample(feature_file_list, label_id, sample_ratio)

        self.feature_file_path = feature_file_path
        self.feature_file_list = feature_file_list
        self.feature_file_extension = feature_file_extension
        self.label_id = label_id
        self.transpose = transpose
        self.feat_length = feat_length
        self.num_channels = num_channels
        self.keep_in_memory = keep_in_memory
        self.transformer = feat_transformer_fn
        self.normalization = normalization

        self.use_label = False if label_id is None else True
        self.transform = False if feat_transformer_fn is None else True
        self.num_obj = len(feature_file_list)

        # logging.info('Feature dir : ' + feature_file_path)

        if self.keep_in_memory:
            logging.info('Loading feature ...... ')
            self._load_all_()
            logging.info('ASVSpoof19Dataset : shape = {}  type : {}'.format(self.feature.shape, self.feature.dtype))

    def _load_all_(self):
        tempfilename = os.path.join(self.feature_file_path, self.feature_file_list[0] + self.feature_file_extension)
        tempfeature = self._loadone_(tempfilename)
        feat_size = tempfeature.shape

        self.feature = np.zeros((self.num_obj, *feat_size), dtype=self.dtype)
        for idx, file in enumerate(self.feature_file_list):
            filename = os.path.join(self.feature_file_path, file + self.feature_file_extension)
            data = self._loadone_(filename)
            self.feature[idx] = data
        self.feature = torch.from_numpy(self.feature)

    def __len__(self):
        return self.num_obj

    def _loadone_(self, filename):
        with h5py.File(filename, 'r') as f:
            data = f['data'][:]
            if data.ndim == 1:
                data = data[:, numpy.newaxis]

        if self.transpose:
            data = data.T

        if self.feat_length > 0:
            data = feature_trim_pad(data, self.feat_length)

        if self.transform:
            data = self.transformer(data)

        if self.normalization:
            data = (data - numpy.mean(data)) / numpy.std(data)

        return data.astype(self.dtype)

    def __getitem__(self, idx):
        if self.keep_in_memory:
            data = self.feature[idx]
        else:

            # fn = self.feature_file_list[idx]
            # if fn == 'LA_D_1340425' or fn == 'LA_D_3295687' or fn == 'LA_D_4192839' or fn == 'LA_D_4697088':
            #     logging.info(fn)

            filename = os.path.join(self.feature_file_path, self.feature_file_list[idx] + self.feature_file_extension)
            data = self._loadone_(filename)
            data = torch.from_numpy(data)

            # fn = self.feature_file_list[idx]
            # if fn == 'LA_D_1340425' or fn == 'LA_D_3295687' or fn =='LA_D_4192839' or fn ==  'LA_D_4697088':
            #     logging.info(fn + str(data.shape))

        if self.use_label:
            return data, self.label_id[idx], idx
        else:
            return data, idx


class ASVspoofWaveDataset(ASVSpoofDataset):
    def __init__(self, feature_file_path, feature_file_list, feature_file_extension='.flac', label_id=None,
                 feat_length=64600,
                 num_channels=0, transpose=False, keep_in_memory=False, sample_ratio=1.0, feat_transformer_fn=None,
                 dtype=np.float32):
        super(ASVspoofWaveDataset, self).__init__(feature_file_path, feature_file_list,
                                                  feature_file_extension=feature_file_extension,
                                                  label_id=label_id,
                                                  feat_length=feat_length, num_channels=num_channels,
                                                  transpose=transpose, keep_in_memory=keep_in_memory,
                                                  sample_ratio=sample_ratio, feat_transformer_fn=feat_transformer_fn,
                                                  dtype=dtype)

    def _loadone_(self, filename):
        # data, sr = librosa.load(filename, sr=16000)
        data, sr = soundfile.read(filename)

        if self.feat_length > 0:
            data = feature_trim_pad(data, self.feat_length)

        return data


class ASVspoofWaveAugmentDataset(ASVSpoofDataset):
    def __init__(self, feature_file_path, feature_file_list, feature_file_extension='.flac', label_id=None,
                 feat_length=64600,
                 num_channels=0, transpose=False, keep_in_memory=False, sample_ratio=1.0, feat_transformer_fn=None,
                 dtype=np.float32, aug_methods=None):
        super(ASVspoofWaveAugmentDataset, self).__init__(feature_file_path, feature_file_list,
                                                         feature_file_extension=feature_file_extension,
                                                         label_id=label_id,
                                                         feat_length=feat_length,
                                                         num_channels=num_channels,
                                                         transpose=transpose,
                                                         keep_in_memory=keep_in_memory,
                                                         sample_ratio=sample_ratio,
                                                         feat_transformer_fn=feat_transformer_fn,
                                                         dtype=dtype)

        self.feature_file_path = '/home/labuser/ssd/lzc/ASVspoof/DS_10283_3336/LA/ASVspoof2019_LA_train/flac'
        logging.info('wave path : ' + self.feature_file_path)

        # self.augmenter = AugmentWAV(
        #     musan_path='/home/lzc/lzc/voxceleb/musan_split',
        #     rir_path='/home/lzc/lzc/voxceleb/RIRS_NOISES/simulated_rirs',
        #     max_frames=400)

    def __getitem__(self, idx):
        filename = os.path.join(self.feature_file_path, self.feature_file_list[idx] + '.flac')
        wave, sr = torchaudio.load(filename)  # , normalize=True

        augmented_audio = self._augment_wave(wave, sr)

        data = extract_feature(augmented_audio, sr, 'LFCC21NN', 'LA')

        data = data.squeeze(0).detach().numpy().T

        if self.feat_length > 0:
            data = feature_trim_pad(data, self.feat_length)

        # if self.transpose:
        #     data = data.T
        #
        # if self.transform:
        #     data = self.transformer(data)

        data = data.astype(self.dtype)
        data = torch.from_numpy(data)

        if self.use_label:
            return data, self.label_id[idx], idx
        else:
            return data, idx

    def _augment_wave(self, audio, sr):

        augtype = random.randint(0, 1)

        if augtype == 0:
            augmented_audio = audio
        elif augtype == 1:
            augmented_audio = augment_audio(audio, sr, 'rb4')
        # elif augtype == 1:
        #     augmented_audio = self.augmenter.reverberate(audio)
        # elif augtype == 2:
        #     augmented_audio = self.augmenter.additive_noise('music', audio)
        # elif augtype == 3:
        #     augmented_audio = self.augmenter.additive_noise('speech', audio)
        # elif augtype == 4:
        #     augmented_audio = self.augmenter.additive_noise('noise', audio)

        return augmented_audio
