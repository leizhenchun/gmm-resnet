########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging
from typing import Optional

import librosa
import numpy as np
import torch
import torchaudio.transforms
from torchaudio import transforms
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import ComputeDeltas

# from .as_lfcc import lfcc, deltas, lfb
from .as_lfcc_baseline import LFCC

# from .as_lfcc_bp import lfcc_bp

transform_vad = None
lfcc_extractor = None


def extract_feature(audio, sr, feature_type, access_type=None, extractor=None):
    # audio, sr = librosa.load(wav_path, sr=16000)
    # audio, sr = soundfile.read(wav_path, dtype='float32')
    # sample_rate = 16000

    global transform_vad
    global lfcc_extractor

    if feature_type == "IMFCC":
        return extract_imfcc(audio, sr)
    elif feature_type == "MFCC":
        return extract_mfcc(audio, sr)
    elif feature_type == "CQT":
        return extract_cqt(audio, sr)
    elif feature_type == "LCQT":
        return extract_log_cqt(audio, sr)

    # elif feature_type == "LSPEC":
    #     return extract_log_spect(audio, n_fft=512, win_length=window_len, hop_length=window_inc)
    # elif feature_type == "LSPECPT":
    #     return extract_log_spect_pt(audio, n_fft=512, win_length=window_len, hop_length=window_inc)
    #
    # elif feature_type == "SPEC":
    #     return extract_spect(audio, n_fft=512, win_length=window_len, hop_length=window_inc)
    # elif feature_type == "SPECPT":
    #     return extract_spect_pt(audio, n_fft=512, win_length=window_len, hop_length=window_inc)

    elif feature_type == 'LFCC':
        return extract_lfcc_pt(audio, sr, int(16000 * 0.02), int(16000 * 0.01), f_min=0, f_max=4000)

    # elif feature_type == 'LFCC19':
    #     # ASVspoof 2019 default settings
    #     return extract_lfcc(audio, sr=sr, win_len=0.020, hop_len=0.010, f_min=0, f_max=8000, nfft=512,
    #                         nfilts=20, num_ceps=20, order_deltas=2, pre_emph=1, delta_norm=False)

    # elif feature_type == 'LFCCBP':
    #     if access_type == 'LA' or access_type == 'DF':
    #         return extract_lfcc_bp(audio, sr=sr, win_len=0.020, hop_len=0.010, f_min=0, f_max=4000, nfft=1024,
    #                                nfilts=70, num_ceps=19, order_deltas=2)
    #     elif access_type == 'PA':
    #         return extract_lfcc_bp(audio, sr=sr, win_len=0.020, hop_len=0.010, f_min=0, f_max=8000, nfft=1024,
    #                                nfilts=70, num_ceps=19, order_deltas=2)
    #
    # elif feature_type == 'LFCC57':
    #     if access_type == 'LA' or access_type == 'DF':
    #         return extract_lfcc(audio, sr=sr, win_len=0.020, hop_len=0.010, f_min=0, f_max=4000, nfft=1024,
    #                             nfilts=70, num_ceps=19, order_deltas=2)
    #     elif access_type == 'PA':
    #         return extract_lfcc(audio, sr=sr, win_len=0.020, hop_len=0.010, f_min=0, f_max=8000, nfft=1024,
    #                             nfilts=70, num_ceps=19, order_deltas=2)
    #
    # elif feature_type == 'LFCC60':
    #     if access_type == 'LA' or access_type == 'DF':
    #         return extract_lfcc(audio, sr=sr, win_len=0.020, hop_len=0.010, f_min=0, f_max=4000, nfft=1024,
    #                             nfilts=70, num_ceps=20, order_deltas=2)
    #     elif access_type == 'PA':
    #         return extract_lfcc(audio, sr=sr, win_len=0.020, hop_len=0.010, f_min=0, f_max=8000, nfft=1024,
    #                             nfilts=70, num_ceps=20, order_deltas=2)
    #
    # elif feature_type == 'LFCC2130':  # for GMM
    #     if access_type == 'LA' or access_type == 'DF':
    #         return extract_lfcc(audio, sr=sr, win_len=0.030, hop_len=0.015, f_min=0, f_max=4000, nfft=1024,
    #                             nfilts=70, num_ceps=19, order_deltas=2)
    #     elif access_type == 'PA':
    #         return extract_lfcc(audio, sr=sr, win_len=0.030, hop_len=0.015, f_min=0, f_max=8000, nfft=1024,
    #                             nfilts=70, num_ceps=19, order_deltas=2)

    # elif feature_type == 'LFCCE21NN':
    #     window_len = int(20 * sr / 1000)
    #     window_inc = int(10 * sr / 1000)
    #
    #     if lfcc_lcnn_extractor == None:
    #         # fl = 320, fs = 160, fn = 1024, sr = sr, filter_num = 20, with_energy = True, max_freq = 0.5
    #         lfcc_lcnn_extractor = LFCC(fl=window_len, fs=window_inc, fn=1024, sr=sr, filter_num=20,
    #                                    with_energy=True, with_emphasis=True,
    #                                    with_delta=True, flag_for_LFB=False,
    #                                    num_coef=None, min_freq=0.0, max_freq=0.5)
    #     return lfcc_lcnn_extractor(audio)

    # return extract_lfcc_lcnn_baseline(sig=audio, sr=sr, win_len=window_len, hop_len=window_inc, f_min=0, f_max=0.5,
    #                                   num_ceps=20, with_energy=True)

    elif feature_type == 'LFCC19':
        if lfcc_extractor == None:
            window_len = int(20 * sr / 1000)
            window_inc = int(10 * sr / 1000)

            max_freq = 1.0

            # fl = 320, fs = 160, fn = 1024, sr = sr, filter_num = 20, with_energy = True, max_freq = 0.5
            lfcc_extractor = LFCC(fl=window_len, fs=window_inc, fn=512, sr=sr, filter_num=20,
                                  with_energy=False, with_emphasis=True,
                                  with_delta=True, flag_for_LFB=False,
                                  num_coef=None, min_freq=0.0, max_freq=max_freq)

        return lfcc_extractor(audio)

    elif feature_type == 'LFCC21NN' or feature_type == 'LFCC21NN_VAD':

        if lfcc_extractor == None:
            window_len = int(20 * sr / 1000)
            window_inc = int(10 * sr / 1000)

            max_freq = 1 if (access_type.lower() == 'pa') else 0.5

            # fl = 320, fs = 160, fn = 1024, sr = sr, filter_num = 20, with_energy = True, max_freq = 0.5
            lfcc_extractor = LFCC(fl=window_len, fs=window_inc, fn=512, sr=sr, filter_num=20,
                                  with_energy=False, with_emphasis=True,
                                  with_delta=True, flag_for_LFB=False,
                                  num_coef=None, min_freq=0.0, max_freq=max_freq)

        return lfcc_extractor(audio)
        # return extract_lfcc_lcnn_baseline(sig=audio, sr=sr, win_len=window_len, hop_len=window_inc, f_min=0, f_max=0.5,
        #                                   num_ceps=20, with_energy=False)

    elif feature_type == 'LFCCPT':

        if lfcc_extractor == None:
            window_len = int(20 * sr / 1000)
            window_inc = int(10 * sr / 1000)

            max_freq = 1 if (access_type.lower() == 'pa') else 0.5

            lfcc_extractor = LFCCPT(sample_rate=sr,
                                    n_filter=20,
                                    f_min=0.0,
                                    f_max=max_freq * float(sr // 2),
                                    n_lfcc=20,
                                    n_fft=512,
                                    win_length=window_len,
                                    hop_length=window_inc)

            # fl = 320, fs = 160, fn = 1024, sr = sr, filter_num = 20, with_energy = True, max_freq = 0.5
            # lfcc_lcnn_extractor = LFCC(fl=window_len, fs=window_inc, fn=1024, sr=sr, filter_num=20,
            #                            with_energy=False, with_emphasis=True,
            #                            with_delta=True, flag_for_LFB=False,
            #                            num_coef=None, min_freq=0.0, max_freq=max_freq)

        return lfcc_extractor(audio)
        # elif feature_type == 'LFCC21NN_VAD':
        #     window_len = int(20 * sr / 1000)
        #     window_inc = int(10 * sr / 1000)
        #
        #     if transform_vad is None:
        #         transform_vad = transforms.Vad(sample_rate=sr, trigger_level=7.5)
        #     if lfcc_lcnn_extractor is None:
        #         # fl = 320, fs = 160, fn = 1024, sr = sr, filter_num = 20, with_energy = True, max_freq = 0.5
        #         lfcc_lcnn_extractor = LFCC(fl=window_len, fs=window_inc, fn=1024, sr=sr, filter_num=20,
        #                                    with_energy=True, with_emphasis=True,
        #                                    with_delta=True, flag_for_LFB=False,
        #                                    num_coef=None, min_freq=0.0, max_freq=0.5)
        #
        #     waveform = transform_vad(audio)
        #     waveform_reversed, _ = apply_effects_tensor(waveform, sr, [["reverse"]])
        #     waveform_reversed = transform_vad(waveform_reversed)
        #     waveform, _ = apply_effects_tensor(waveform_reversed, sr, [["reverse"]])
        #
        #     return lfcc_lcnn_extractor(waveform)

        # return extract_lfcc_lcnn_baseline_vad(sig=audio, sr=sr, win_len=window_len, hop_len=window_inc, f_min=0,
        #                                       f_max=0.5,
        #                                       num_ceps=20, with_energy=False)

    # elif feature_type == 'LFB':
    #     return extract_lfb_pt(audio, sr=sr, win_len=0.030, hop_len=0.010, f_min=0, f_max=8000, num_ceps=60,
    #                           order_deltas=0)

    elif feature_type == 'MSPEC':
        return extract_mel_spect(audio, n_fft=512, win_length=512, hop_length=128, n_mels=60)

    elif feature_type == 'WAVE_VAD':
        if transform_vad is None:
            transform_vad = transforms.Vad(sample_rate=sr, trigger_level=7.5)

        waveform = transform_vad(audio)
        waveform_reversed, _ = apply_effects_tensor(waveform, sr, [["reverse"]])
        waveform_reversed = transform_vad(waveform_reversed)
        waveform, _ = apply_effects_tensor(waveform_reversed, sr, [["reverse"]])

        return waveform
    else:
        raise ValueError('feature_type error:' + feature_type)


def trim_silence(audio, threshold=0.1, frame_length=2048):
    if audio.size < frame_length:
        frame_length = audio.size
    # energy = librosa.feature.rmse(audio, frame_length=frame_length)
    energy = librosa.feature.rms(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def extract_imfcc(audio, sr, n_imfcc, n_fft, hop_length):
    S = np.abs(librosa.core.stft(audio, n_fft=n_fft, hop_length=hop_length)) ** 2.0
    mel_basis = librosa.filters.mel(sr, n_fft)
    mel_basis = np.linalg.pinv(mel_basis).T
    mel = np.dot(mel_basis, S)
    S = librosa.power_to_db(mel)
    imfcc = np.dot(librosa.filters.dct(n_imfcc, S.shape[0]), S)
    imfcc_delta = librosa.feature.delta(imfcc)
    imfcc_delta_delta = librosa.feature.delta(imfcc)
    feature = np.concatenate((imfcc, imfcc_delta, imfcc_delta_delta), axis=0)
    return feature


def extract_mfcc(audio, sr, n_mfcc, n_fft, hop_length):
    y = audio
    if y.size == 0:
        y = audio
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_delta = librosa.feature.delta(mfcc)
    feature = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta), axis=0)
    return feature


def extract_cqt(audio, sr, n_cqt, f_min, hop_length):
    y = trim_silence(audio)
    if y.size == 0:
        y = audio
    cqt = librosa.feature.chroma_cqt(y, sr, hop_length=hop_length, fmin=f_min, n_chroma=n_cqt, n_octaves=5)
    return cqt


def extract_log_cqt(audio, sr, f_min, hop_length):
    cqt = librosa.feature.chroma_cqt(audio, sr, hop_length=hop_length, fmin=f_min, n_chroma=84, n_octaves=7)
    return librosa.amplitude_to_db(cqt)


def extract_spect(audio, n_fft, win_length, hop_length):
    result, _ = librosa.core.spectrum._spectrogram(audio, n_fft=n_fft, hop_length=hop_length, power=2,
                                                   win_length=win_length,
                                                   window="hann")
    result = result[0:-1, :]
    return result


def extract_log_spect(audio, n_fft, win_length, hop_length):
    S, _ = librosa.core.spectrum._spectrogram(audio, n_fft=n_fft, hop_length=hop_length, power=2,
                                              win_length=win_length,
                                              window="hann")
    result = librosa.power_to_db(S)
    result = result[0:-1, :]
    return result


def extract_mel_spect(audio, n_fft=512, win_length=512, hop_length=128, n_mels=60):
    # transform = MelSpectrogram(sample_rate=16000, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
    #                            n_mels=n_mels)
    # result = transform(audio)
    # librosa.filters.mel
    result = librosa.feature.melspectrogram(y=audio,
                                            sr=16000,
                                            S=None,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            win_length=win_length,
                                            window="hann",
                                            center=True,
                                            pad_mode="reflect",
                                            power=2.0,
                                            n_mels=n_mels)
    return result


# def extract_spect_pt(audio, n_fft, win_length, hop_length):
#     transform = Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, )
#     result = transform(torch.from_numpy(audio)).numpy()
#     result = result[0:-1, :]
#     return result
#
#
# def extract_log_spect_pt(audio, n_fft, win_length, hop_length):
#     transform = Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, )
#     amp2db = AmplitudeToDB()
#     result = amp2db(transform(torch.from_numpy(audio))).numpy()
#     result = result[0:-1, :]
#     return result

# def extract_lfcc_baseline(sig, sr):
#     wav = torch.Tensor(np.expand_dims(sig, axis=0))
#     extractor = LFCC(fl=320, fs=160, fn=1024, sr=sr, filter_num=20, with_energy=True, max_freq=0.5)
#     lfcc = extractor(wav)
#     return lfcc.transpose(1, 2).squeeze(axis=0)

# def extract_lfcc(audio, sr, win_len, hop_len, f_min=0, f_max=4000, num_ceps=20, nfilts=70, nfft=1024,
#                  order_deltas=2,
#                  pre_emph=0, delta_norm=True):
#     lfccs = lfcc(sig=audio,
#                  fs=sr,
#                  num_ceps=num_ceps,
#                  pre_emph=pre_emph,
#                  win_len=win_len,  # 0.030,
#                  win_hop=hop_len,  # 0.015,
#                  nfilts=nfilts,
#                  nfft=nfft,
#                  low_freq=f_min,
#                  high_freq=f_max).T
#     if order_deltas > 0:
#         feats = list()
#         feats.append(lfccs)
#         for d in range(order_deltas):
#             feats.append(deltas(feats[-1], width=3, norm=delta_norm))
#         lfccs = vstack(feats)
#     return lfccs


# def extract_lfcc_bp(audio, sr, win_len, hop_len, f_min=0, f_max=4000, num_ceps=20, nfilts=70, nfft=1024,
#                     order_deltas=2,
#                     pre_emph=0, delta_norm=True):
#     lfccs = lfcc_bp(sig=audio,
#                     fs=sr,
#                     num_ceps=num_ceps,
#                     pre_emph=pre_emph,
#                     win_len=win_len,  # 0.030,
#                     win_hop=hop_len,  # 0.015,
#                     nfilts=nfilts,
#                     nfft=nfft,
#                     low_freq=f_min,
#                     high_freq=f_max).T
#     if order_deltas > 0:
#         feats = list()
#         feats.append(lfccs)
#         for d in range(order_deltas):
#             feats.append(deltas(feats[-1], width=3, norm=delta_norm))
#         lfccs = vstack(feats)
#     return lfccs


class MyLFCC:
    pass


def extract_lfcc_pt(audio, sr, win_len, hop_len, f_min=0, f_max=4000):
    speckwargs = {}
    speckwargs['n_fft'] = 512
    speckwargs['win_length'] = win_len
    speckwargs['hop_length'] = hop_len
    # speckwargs['center'] = False
    # speckwargs['pad_mode'] = 'constant'

    # sample_rate: int = 16000,
    # n_filter: int = 128,
    # f_min: float = 0.0,
    # f_max: Optional[float] = None,
    # n_lfcc: int = 40,
    # dct_type: int = 2,
    # norm: str = "ortho",
    # log_lf: bool = False,
    # speckwargs: Optional[dict] = None,

    extractor = torchaudio.transforms.LFCC(sample_rate=sr, f_min=f_min, f_max=f_max, n_lfcc=20,
                                           speckwargs=speckwargs)
    delta = ComputeDeltas()

    audio = torch.from_numpy(audio)
    lfcc = extractor(audio)
    lfcc_d = delta(lfcc)
    lfcc_dd = delta(lfcc_d)
    result = torch.cat([lfcc, lfcc_d, lfcc_dd], dim=0)
    return result.numpy()


# def extract_lfb_pt(sig, sr, win_len, hop_len, f_min, f_max, num_ceps=60, order_deltas=0):
#     lfb_feat = lfb(sig=sig,
#                    fs=sr,
#                    num_ceps=num_ceps,
#                    win_len=win_len,
#                    win_hop=hop_len,
#                    low_freq=f_min,
#                    high_freq=f_max).T
#
#     if order_deltas > 0:
#         feats = list()
#         feats.append(lfb_feat)
#         for d in range(order_deltas):
#             feats.append(deltas(feats[-1]))
#         lfb_feat = vstack(feats)
#     return lfb_feat


# def extract_lfcc_lcnn_baseline(sig, sr, win_len, hop_len, f_min=0, f_max=0.5, num_ceps=20, with_energy=True,
#                                flag_for_LFB=False):
#     sig = sig.squeeze().detach().numpy()
#     # augmented_audio = augmented_audio.astype(numpy.float32)
#
#     sig = sig.astype(numpy.float32)
#
#     # fl = 320, fs = 160, fn = 1024, sr = sr, filter_num = 20, with_energy = True, max_freq = 0.5
#     lfcc_extractor = LFCC(fl=win_len, fs=hop_len, fn=1024, sr=sr, filter_num=num_ceps,
#                           with_energy=with_energy, with_emphasis=True,
#                           with_delta=True, flag_for_LFB=flag_for_LFB,
#                           num_coef=None, min_freq=f_min, max_freq=f_max)
#     sig = torch.from_numpy(sig).unsqueeze(0)
#     lfcc = lfcc_extractor(sig)
#     lfcc = lfcc.squeeze().detach().numpy().T
#
#     return lfcc

def extract_lfcc_lcnn_baseline(sig, sr, win_len, hop_len, f_min=0, f_max=0.5, num_ceps=20, with_energy=True,
                               flag_for_LFB=False):
    # sig = sig.squeeze().detach().numpy()
    # augmented_audio = augmented_audio.astype(numpy.float32)

    # sig = sig.astype(numpy.float32)

    global lfcc_extractor
    if lfcc_extractor == None:
        # fl = 320, fs = 160, fn = 1024, sr = sr, filter_num = 20, with_energy = True, max_freq = 0.5
        lfcc_extractor = LFCC(fl=win_len, fs=hop_len, fn=1024, sr=sr, filter_num=num_ceps,
                              with_energy=with_energy, with_emphasis=True,
                              with_delta=True, flag_for_LFB=flag_for_LFB,
                              num_coef=None, min_freq=f_min, max_freq=f_max)
    # sig = torch.from_numpy(sig).unsqueeze(0)
    lfcc = lfcc_extractor(sig)
    # lfcc = lfcc.squeeze().detach().numpy().T

    return lfcc


def extract_lfcc_lcnn_baseline_vad(sig, sr, win_len, hop_len, f_min=0, f_max=0.5, num_ceps=20, with_energy=True,
                                   flag_for_LFB=False):
    # waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
    global transform_vad
    if transform_vad == None:
        transform_vad = transforms.Vad(sample_rate=sr, trigger_level=7.5)

    waveform = transform_vad(sig)
    waveform_reversed, sample_rate = apply_effects_tensor(waveform, sr, [["reverse"]])
    waveform_reversed_front_trim = transform_vad(waveform_reversed)
    sig, sample_rate = apply_effects_tensor(waveform_reversed_front_trim, sample_rate, [["reverse"]])

    return extract_lfcc_lcnn_baseline(sig, sr, win_len, hop_len, f_min=f_min, f_max=f_max, num_ceps=num_ceps,
                                      with_energy=with_energy, flag_for_LFB=flag_for_LFB)


class LFCCPT(torch.nn.Module):
    def __init__(self,
                 sample_rate: int = 16000,
                 n_filter: int = 128,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 n_lfcc: int = 40,
                 n_fft=512,
                 win_length=320,
                 hop_length=160,
                 ) -> None:
        super(LFCCPT, self).__init__()
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.n_filter = n_filter
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.preemphasis = torchaudio.transforms.Preemphasis()
        self.lfcc = torchaudio.transforms.LFCC(sample_rate=self.sample_rate,
                                               n_filter=self.n_filter,
                                               f_min=self.f_min,
                                               f_max=self.f_max,
                                               n_lfcc=self.n_lfcc,
                                               dct_type=2,
                                               norm="ortho",
                                               log_lf=True,
                                               speckwargs={"n_fft": self.n_fft,
                                                           "win_length": self.win_length,
                                                           "hop_length": self.hop_length,
                                                           "window_fn": torch.hamming_window,
                                                           "pad_mode": "constant",
                                                           }, )

        self.delta = torchaudio.transforms.ComputeDeltas(win_length=3)

    def forward(self, waveform):
        waveform = self.preemphasis(waveform)
        lfcc = self.lfcc(waveform)
        delta = self.delta(lfcc)
        delta2 = self.delta(delta)

        return torch.cat((lfcc, delta, delta2), dim=1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.NOTSET, format="[%(asctime)s - %(levelname)1.1s] %(message)s")
