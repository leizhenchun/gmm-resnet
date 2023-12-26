########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import glob
import os
import random

import numpy
import soundfile
import torch
import torchaudio
from torch import Tensor
from torchaudio.functional import apply_codec

from augmentation.RawBoost import LnL_convolutive_noise, ISD_additive_noise, SSI_additive_noise, normWav

# from asvspoof21.torchaudio_util import print_stats

if os.name == 'posix':
    rirs_path = r'/home/lzc/lzc/RIRS_NOISES/simulated_rirs'
    musan_path = r'/home/lzc/lzc/musan'
    path_sep = '/'
elif os.name == 'nt':
    rirs_path = r'e:\RIRS_NOISES\simulated_rirs'
    musan_path = r'e:\musan'
    path_sep = '\\'

noise_num_range = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
noise_snr_range = {'noise': [5, 20], 'speech': [5, 20], 'music': [5, 20]}
noise_files = {'noise': [], 'speech': [], 'music': []}

rir_files = glob.glob(os.path.join(rirs_path, '*/*/*.wav'))
musan_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
for file in musan_files:
    noise_files[file.split(path_sep)[-3]].append(file)


def augment_audio(audio: Tensor, sr, aug_method):
    aug_method = aug_method.lower()

    if aug_method == 'original':
        return audio

    elif aug_method == 'alaw':
        return apply_codec(audio, sr, format='wav', encoding='ALAW')

    elif aug_method == 'ulaw':
        return apply_codec(audio, sr, format='wav', encoding='ULAW')

    elif aug_method == 'gsm':
        return apply_codec(audio, sr, format='gsm')

    elif aug_method == 'mp3':
        return apply_codec(audio, sr, format='mp3')

    elif aug_method == 'vorbis':
        return apply_codec(audio, sr, format='vorbis')

    elif aug_method == 'rir':
        return apply_rir(audio, sr)

    elif aug_method in ['noise', 'music', 'speech']:
        return add_noise(audio, aug_method)

    elif aug_method == 'rb1':
        audio = audio.squeeze().numpy()
        augmented_audio = apply_rb1(audio, sr)
        augmented_audio = torch.from_numpy(augmented_audio.astype(numpy.float32)).unsqueeze(0)
        return augmented_audio

    elif aug_method == 'rb2':
        audio = audio.squeeze().numpy()
        augmented_audio = apply_rb2(audio)
        augmented_audio = torch.from_numpy(augmented_audio.astype(numpy.float32)).unsqueeze(0)
        return augmented_audio

    elif aug_method == 'rb3':
        audio = audio.squeeze().numpy()
        augmented_audio = apply_rb3(audio, sr)
        augmented_audio = torch.from_numpy(augmented_audio.astype(numpy.float32)).unsqueeze(0)
        return augmented_audio

    elif aug_method == 'rb4':  # together in series (1+2+3)
        audio = audio.squeeze().numpy()
        augmented_audio = apply_rb1(audio, sr)
        augmented_audio = apply_rb2(augmented_audio)
        augmented_audio = apply_rb3(augmented_audio, sr)
        augmented_audio = torch.from_numpy(augmented_audio.astype(numpy.float32)).unsqueeze(0)
        return augmented_audio

    elif aug_method in ['rb5', 'rb51', 'rb52', 'rb53']:  # together in series (1+2)
        audio = audio.squeeze().numpy()
        augmented_audio = apply_rb1(audio, sr)
        augmented_audio = apply_rb2(augmented_audio)
        augmented_audio = torch.from_numpy(augmented_audio.astype(numpy.float32)).unsqueeze(0)
        augmented_audio = augmented_audio.unsqueeze(0)
        return augmented_audio

    elif aug_method == 'rb6':  # together in series (1+3)
        audio = audio.squeeze().numpy()
        augmented_audio = apply_rb1(audio, sr)
        augmented_audio = apply_rb3(augmented_audio, sr)
        augmented_audio = torch.from_numpy(augmented_audio.astype(numpy.float32)).unsqueeze(0)
        return augmented_audio

    elif aug_method == 'rb7':  # together in series (2+3)
        audio = audio.squeeze().numpy()
        augmented_audio = apply_rb2(audio)
        augmented_audio = apply_rb3(augmented_audio, sr)
        augmented_audio = torch.from_numpy(augmented_audio.astype(numpy.float32)).unsqueeze(0)
        return augmented_audio

    elif aug_method == 'rb8':  # together in Parallel (1||2)
        audio = audio.squeeze().numpy()
        augmented_audio1 = apply_rb1(audio, sr)
        augmented_audio2 = apply_rb2(audio)

        augmented_audio = augmented_audio1 + augmented_audio2
        augmented_audio = normWav(augmented_audio, 0)
        augmented_audio = torch.from_numpy(augmented_audio.astype(numpy.float32)).unsqueeze(0)
        return augmented_audio

    # return augmented_audio


def apply_rir(audio, sr):
    rir, sr = soundfile.read(random.choice(rir_files))

    if audio.dtype == torch.float32:
        rir = rir.astype(numpy.float32)

    # rir = numpy.expand_dims(rir, 0)
    # rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
    # augmented = signal.convolve(audio, rir, mode='full')[:, :audio.shape[1]]
    # augmented = torch.from_numpy(augmented)

    # rir = rir_raw[:, int(sample_rate * 1.01): int(sample_rate * 1.3)]
    rir = torch.from_numpy(rir).unsqueeze(0)
    rir = rir / torch.norm(rir, p=2)
    rir = torch.flip(rir, [1])
    speech_ = torch.nn.functional.pad(audio, (rir.shape[1] - 1, 0))
    augmented = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]

    return augmented


def add_noise(audio, noise_type):
    audio = audio.numpy()
    clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
    noise_num = noise_num_range[noise_type]
    selected_noise_files = random.sample(noise_files[noise_type], random.randint(noise_num[0], noise_num[1]))
    noises = []
    for noise_file in selected_noise_files:
        noise, sr = soundfile.read(noise_file)
        length = audio.shape[1]
        if noise.shape[0] <= length:
            shortage = length - noise.shape[0]
            noise = numpy.pad(noise, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (noise.shape[0] - length))
        noise = noise[start_frame:start_frame + length]
        noise = numpy.stack([noise], axis=0)
        noise_db = 10 * numpy.log10(numpy.mean(noise ** 2) + 1e-4)
        noise_snr = random.uniform(noise_snr_range[noise_type][0], noise_snr_range[noise_type][1])
        noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noise)
    noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
    augmented = noise + audio
    augmented = torch.from_numpy(augmented)

    return augmented


def add_noise2(audio, noise_type):
    # audio = audio.numpy()
    # clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
    audio_power = audio.norm(p=2)
    audio_length = audio.shape[1]

    noise_num = noise_num_range[noise_type]
    selected_noise_files = random.sample(noise_files[noise_type], random.randint(noise_num[0], noise_num[1]))
    noises = []
    for noise_file in selected_noise_files:
        noise, sr = soundfile.read(noise_file)

        if noise.shape[0] <= audio_length:
            shortage = audio_length - noise.shape[0]
            noise = numpy.pad(noise, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (noise.shape[0] - audio_length))
        noise = noise[start_frame:start_frame + audio_length]

        noise_power = noise.norm(p=2)
        noise = numpy.stack([noise], axis=0)
        noise_snr = random.uniform(noise_snr_range[noise_type][0], noise_snr_range[noise_type][1])
        snr = 10 ** (noise_snr / 20)
        scale = snr * noise_power / audio_power
        noises.append()
        # noisy_speeches.append((scale * speech + noise) / 2)

        # noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noise)
    noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
    augmented = noise + audio
    augmented = torch.from_numpy(augmented)

    return augmented  # .squeeze(0)


def apply_rb1(audio, sr):
    augmented_audio = LnL_convolutive_noise(audio, N_f=5, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
                                            minCoeff=10, maxCoeff=100, minG=0, maxG=0,
                                            minBiasLinNonLin=5, maxBiasLinNonLin=20, fs=sr)
    return augmented_audio


def apply_rb2(audio):
    augmented_audio = ISD_additive_noise(audio, P=10, g_sd=2)
    return augmented_audio


def apply_rb3(audio, sr):
    augmented_audio = SSI_additive_noise(audio, SNRmin=10, SNRmax=40, nBands=5, minF=20, maxF=8000, minBW=100,
                                         maxBW=1000, minCoeff=10, maxCoeff=100, minG=0, maxG=0, fs=sr)
    return augmented_audio


if __name__ == '__main__':
    # audio_file = r'E:\flac\LA_T_1004407.flac'
    audio_file = r'/home/lzc/lzc/ASVspoof2019/DS_10283_3336/LA/ASVspoof2019_LA_train/flac/LA_T_1004407.flac'
    # audio1, sr1 = soundfile.read(audio_file)
    audio, sr = torchaudio.load(audio_file)
    # print_stats(audio, sr)
    # plot_specgram(audio, sr)

    # torchaudio.save('d:\\aaa.wav', audio, sr)
    #
    # audio2 = augment_audio(audio, sr, 'rir')
    # torchaudio.save('d:\\bbb.wav', audio2, sr)
    #
    # audio3 = augment_audio(audio, sr, 'noise')
    # torchaudio.save('d:\\ccc.wav', audio3, sr)
    #
    # audio4 = augment_audio(audio, sr, 'music')
    # torchaudio.save('d:\\ddd.wav', audio4, sr)
    #
    # audio5 = augment_audio(audio, sr, 'speech')
    # torchaudio.save('d:\\eee.wav', audio5, sr)

    # audio6 = augment_audio(audio, sr, 'rb5')
    # torchaudio.save('d:\\eee.wav', audio6, sr)

    audio7 = augment_audio(audio, sr, 'alaw')
    torchaudio.save('d:\\eee.wav', audio7, sr)
