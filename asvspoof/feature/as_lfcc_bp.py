import numpy as np
import scipy
from numpy import floor, tile, concatenate
from scipy.signal import lfilter
from skfuzzy import trimf


def pre_emphasis(sig, pre_emph_coeff=0.97):
    """
    perform preemphasis on the input signal.

    Args:
        sig   (array) : signal to filter.
        coeff (float) : preemphasis coefficient. 0 is no filter, default is 0.95.

    Returns:
        the filtered signal.
    """
    return np.append(sig[0], sig[1:] - pre_emph_coeff * sig[:-1])


def stride_trick(a, stride_length, stride_step):
    """
    apply framing using the stride trick from numpy.

    Args:
        a (array) : signal array.
        stride_length (int) : length of the stride.
        stride_step (int) : stride step.

    Returns:
        blocked/framed array.
    """
    # nrows = ((a.size - stride_length) // stride_step) + 1
    nrows = ((a.size - stride_length) // stride_step) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, stride_length), strides=(stride_step * n, n))


def framing(sig, fs=16000, win_len=0.025, win_hop=0.01):
    """
    transform a signal into a series of overlapping frames (=Frame blocking).

    Args:
        sig     (array) : a mono audio signal (Nx1) from which to compute features.
        fs        (int) : the sampling frequency of the signal we are working with.
                          Default is 16000.
        win_len (float) : window length in sec.
                          Default is 0.025.
        win_hop (float) : step between successive windows in sec.
                          Default is 0.01.

    Returns:
        array of frames.
        frame length.

    Notes:
    ------
        Uses the stride trick to accelerate the processing.
    """
    # run checks and assertions
    if win_len < win_hop:
        raise ValueError("win_len < win_hop")

    # compute frame length and frame step (convert from seconds to samples)
    frame_length = win_len * fs
    frame_step = win_hop * fs

    last_frame_len = len(sig) % frame_step
    if last_frame_len > 0:
        extend_zeros = np.zeros(int(frame_step - last_frame_len), dtype=sig.dtype)
        sig = np.hstack((sig, extend_zeros))

    # signal_length = len(sig)
    # frames_overlap = frame_length - frame_step

    # make sure to use integers as indices
    frames = stride_trick(sig, int(frame_length), int(frame_step))
    if len(frames[-1]) < frame_length:
        frames[-1] = np.append(frames[-1], np.array([0] * (frame_length - len(frames[0]))))

    return frames, frame_length


def windowing(frames, frame_len, win_type="hamming", beta=14):
    """
    generate and apply a window function to avoid spectral leakage.

    Args:
        frames  (array) : array including the overlapping frames.
        frame_len (int) : frame length.
        win_type  (str) : type of window to use.
                          Default is "hamming"

    Returns:
        windowed frames.
    """
    if win_type == "hamming":
        windows = np.hamming(frame_len)
    elif win_type == "hanning":
        windows = np.hanning(frame_len)
    elif win_type == "bartlet":
        windows = np.bartlett(frame_len)
    elif win_type == "kaiser":
        windows = np.kaiser(frame_len, beta)
    elif win_type == "blackman":
        windows = np.blackman(frame_len)
    windowed_frames = frames * windows
    return windowed_frames


def get_linear_fbanks(fs, nfft, No_Filter, low_freq, high_freq):
    f = (fs / 2) * np.linspace(0, 1, int(nfft / 2 + 1))
    filbandwidthsf = np.linspace(low_freq, high_freq, No_Filter + 2)

    closestIndex_low_freq = np.argmin(abs(f - low_freq))
    closestIndex_high_freq = np.argmin(abs(f - high_freq))
    fbank_size = closestIndex_high_freq - closestIndex_low_freq + 1

    filterbank = np.zeros((fbank_size, No_Filter))
    f = f[closestIndex_low_freq:closestIndex_high_freq + 1]

    for i in range(No_Filter):
        filterbank[:, i] = trimf(f, [filbandwidthsf[i], filbandwidthsf[i + 1], filbandwidthsf[i + 2]])

    return filterbank


def linear_fbanks(fr_all, fs, nfft, No_Filter, low_freq, high_freq):
    f = (fs / 2) * np.linspace(0, 1, int(nfft / 2 + 1))
    filbandwidthsf = np.linspace(low_freq, high_freq, No_Filter + 2)

    closestIndex_low_freq = np.argmin(abs(f - low_freq))
    closestIndex_high_freq = np.argmin(abs(f - high_freq))
    fa_all = fr_all[:, closestIndex_low_freq:closestIndex_high_freq + 1]

    filterbank = np.zeros((fa_all.shape[1], No_Filter))
    f = f[closestIndex_low_freq:closestIndex_high_freq + 1]

    for i in range(No_Filter):
        filterbank[:, i] = trimf(f, [filbandwidthsf[i], filbandwidthsf[i + 1], filbandwidthsf[i + 2]])

    filbanksum = np.dot(fa_all, filterbank)

    return filbanksum


def deltas(x, width=3, norm=True):
    hlen = int(floor(width / 2))
    win = list(range(hlen, -hlen - 1, -1))
    xx_1 = tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx)
    D = D[:, hlen * 2:]
    if norm:
        D = D / (2 * sum([i ** 2 for i in range(1, hlen + 1)]))
    return D


def lfcc_bp(sig,
            fs=16000,
            num_ceps=20,
            pre_emph=0,
            pre_emph_coeff=0.97,
            win_len=0.030,
            win_hop=0.015,
            win_type="hamming",
            nfilts=70,
            nfft=1024,
            low_freq=None,
            high_freq=None,
            scale="constant",
            dct_type=2,
            normalize=0):
    """
    Compute the linear-frequency cepstral coefﬁcients (GFCC features) from an audio signal.
    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        win_type       (float) : window type to apply for the windowing.
                                 Default is "hamming".
        nfilts           (int) : the number of filters in the filterbank.
                                 Default is 40.
        nfft             (int) : number of FFT points.
                                 Default is 512.
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 Default is 0.
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 Default is samplerate / 2 = 8000.
        scale           (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                                 Default is "constant".
        dct_type         (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                                 Default is 2.
        use_energy       (int) : overwrite C0 with true log energy
                                 Default is 0.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.
    Returns:
        (array) : 2d array of LFCC features (num_frames x num_ceps)
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ValueError("low_freq < 0")
    if high_freq > (fs / 2):
        raise ValueError("high_freq > (fs / 2)")
    if nfilts < num_ceps:
        raise ValueError("nfilts < num_ceps")

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # -> framing
    frames, frame_length = framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames, frame_len=frame_length, win_type=win_type)

    # -> FFT -> |.|
    fourrier_transform = np.fft.rfft(windows, nfft)
    abs_fft_values = np.abs(fourrier_transform) ** 2

    #  -> x linear-fbanks
    # linear_fbanks_mat = linear_filter_banks(nfilts=nfilts,
    #                                         nfft=nfft,
    #                                         fs=fs,
    #                                         low_freq=low_freq,
    #                                         high_freq=high_freq,
    #                                         scale=scale)
    #
    # features = np.dot(abs_fft_values, linear_fbanks_mat.T)
    # features = linear_fbanks(abs_fft_values, fs, nfft, nfilts, low_freq, high_freq)

    f = (fs / 2) * np.linspace(0, 1, int(nfft / 2 + 1))
    closestIndex_low_freq = np.argmin(abs(f - low_freq))
    closestIndex_high_freq = np.argmin(abs(f - high_freq))
    abs_fft_values = abs_fft_values[:, closestIndex_low_freq:closestIndex_high_freq + 1]
    linear_fbanks_mat = get_linear_fbanks(fs, nfft, nfilts, low_freq, high_freq)
    features = np.dot(abs_fft_values, linear_fbanks_mat)

    log_features = np.log10(features + 2.2204e-16)

    #  -> DCT(.)
    # lfccs = dct(log_features, type=dct_type, norm='ortho', axis=1)[:, :num_ceps]
    lfccs = scipy.fftpack.dct(x=log_features, type=2, axis=1, norm='ortho')[:, :num_ceps]

    return lfccs


def lfb(sig,
        fs=16000,
        num_ceps=20,
        pre_emph=0,
        pre_emph_coeff=0.97,
        win_len=0.030,
        win_hop=0.015,
        win_type="hamming",
        nfilts=70,
        nfft=1024,
        low_freq=None,
        high_freq=None):
    """
    Compute the linear-frequency cepstral coefﬁcients (GFCC features) from an audio signal.
    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        win_type       (float) : window type to apply for the windowing.
                                 Default is "hamming".
        nfilts           (int) : the number of filters in the filterbank.
                                 Default is 40.
        nfft             (int) : number of FFT points.
                                 Default is 512.
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 Default is 0.
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 Default is samplerate / 2 = 8000.
        scale           (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                                 Default is "constant".
        dct_type         (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                                 Default is 2.
        use_energy       (int) : overwrite C0 with true log energy
                                 Default is 0.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.
    Returns:
        (array) : 2d array of LFCC features (num_frames x num_ceps)
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ValueError("low_freq < 0")
    if high_freq > (fs / 2):
        raise ValueError("high_freq > (fs / 2)")
    if nfilts < num_ceps:
        raise ValueError("nfilts < num_ceps")

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # -> framing
    frames, frame_length = framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames, frame_len=frame_length, win_type=win_type)

    # -> FFT -> |.|
    fourrier_transform = np.fft.rfft(windows, nfft)
    abs_fft_values = np.abs(fourrier_transform) ** 2

    linear_fbanks_mat = get_linear_fbanks(fs, nfft, nfilts, low_freq, high_freq)
    features = np.dot(abs_fft_values, linear_fbanks_mat)

    log_features = np.log10(features + 2.2204e-16)

    return log_features
