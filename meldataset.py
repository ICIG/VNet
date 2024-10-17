# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import pathlib
from tqdm import tqdm

MAX_WAV_VALUE = 32768.0


def load_wav(full_path, sr_target):
    sampling_rate, data = read(full_path)
    if sampling_rate != sr_target:
        raise RuntimeError("Sampling rate of the file {} is {} Hz, but the model requires {} Hz".
              format(full_path, sampling_rate, sr_target))
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

#计算音频信号的梅尔谱图的函数
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    #检查输入音频信号的最小值和最大值是否超过了范围（-1到1），如果不在，打印相应的警告信息
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    #检查是否已经计算了给定最高频率fmax的梅尔滤波器，它使用mel_basis字典来储存计算过的梅尔滤波器
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    #代码对输入音频信号进行填充，使其长度与n_fft和hop_size对齐，并将其压缩为一维张量。
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    #使用torch.stft函数计算输入音频信号的短时傅里叶变换。它将结果转换为实数张量，通过对每个复数值的实部和虚部进行平方和开平方，得到STFT的幅度谱。
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    #通过矩阵乘法将幅度谱与梅尔滤波器矩阵相乘，得到梅尔频谱
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    #对梅尔频谱进行归一化处理，以使其在时间轴方向上的统计特性保持一致。
    spec = spectral_normalize_torch(spec)
    #函数返回计算得到的梅尔频谱。
    return spec

#获取数据集中的文件列表
def get_dataset_filelist(a):
    #使用open函数打开训练文件列表文件，并将其赋值给变量fi。文件以只读模式打开，并使用utf-8编码
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        #读取文件的内容，并将每一行通过换行符分割为列表。对于每个非空行，将其与a.input_wavs_dir和.wav后缀拼接起来，形成完整的音频文件路径，这些路径被储存在training_files列表中。
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]
        #打印训练文件列表中的第一个文件路径
        print("first training file: {}".format(training_files[0]))

    #使用open函数打开验证文件列表文件，并将其赋值给变量fi，文件以只读模式打开，并使用utf-8编码。
    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        #读取文件的内容，并将每一行通过换行符分割为列表。对于 每个非空行，将其与a.input_wavs_dir和.wav后缀拼接起来，形成完整的音频文件路径，这些路径被储存在validation_files列表中。
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
        #打印验证文件列表中的第一个文件路径
        print("first validation file: {}".format(validation_files[0]))

    #创建一个空列表list_unseen_validation_files，用于存储未见过的验证文件的路径。通过循环遍历a.list_input_unseen_validation_file列表，获取每个未见过的验证文件路径列表。
    list_unseen_validation_files = []
    for i in range(len(a.list_input_unseen_validation_file)):
        with open(a.list_input_unseen_validation_file[i], 'r', encoding='utf-8') as fi:
            #读取文件的内容，并将每一行通过换行符分割为列表。对于每个非空行，将其与a.list_input_unseen_wavs_dir[i]和.wav后缀拼接起来，形成完整的音频文件路径，这些路径被储存在unseen_validation_files列表中。
            unseen_validation_files = [os.path.join(a.list_input_unseen_wavs_dir[i], x.split('|')[0] + '.wav')
                                for x in fi.read().split('\n') if len(x) > 0]
            #打印当前未见过的验证文件集合中的第一个文件路径，并将整个文件路径列表添加到list_unseen_validation_files列表中。
            print("first unseen {}th validation fileset: {}".format(i, unseen_validation_files[0]))
            list_unseen_validation_files.append(unseen_validation_files)

    #返回三个值：训练文件列表，验证文件列表，未见过的验证文件列表。
    return training_files, validation_files, list_unseen_validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, hparams, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None, is_seen=True):
        #将传入的training_files赋值给self.audio_files，如果shuffle参数为True，则通过随机打乱列表中的元素来对self.audio_files进行洗牌。
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.hparams = hparams
        self.is_seen = is_seen
        #根据is_seen的值，设置self.name的属性，如果is_seen为True，将self.audio_files[0]的路径的第一部分作为self.name，否则将self.audio_files[0]的路径的前两部分连接起来，并使用连字符分割，作为self.name
        if self.is_seen:
            self.name = pathlib.Path(self.audio_files[0]).parts[0]
        else:
            self.name = '-'.join(pathlib.Path(self.audio_files[0]).parts[:2]).strip("/")

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

        #打印信息检查数据集的完整性，对于self.audio_files列表中的每个文件路径，确保该文件存在。
        print("INFO: checking dataset integrity...")
        for i in tqdm(range(len(self.audio_files))):
            assert os.path.exists(self.audio_files[i]), "{} not found".format(self.audio_files[i])

    #定义__getitem__方法，该方法用于从数据集中获取一个样本
    def __getitem__(self, index):

        #获取self.audio_files列表中索引为index的文件路径，并将其赋值给filename
        filename = self.audio_files[index]
        #如果缓存的音频引用计数为0，则加载指定文件路径的音频文件，并进行相应的预处理操作。如果缓存的音频引用计数不为0，则直接使用缓存的音频数据。
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename, self.sampling_rate)
            audio = audio / MAX_WAV_VALUE#将音频值除以MAX_WAV_VALUE
            if not self.fine_tuning:#如果fine_tuning不为True，将音频值进行归一化并乘以0.95
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        #将audio转换为torch.FloatTensor类型，并在第0维添加一个维度
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        #判断fine_tuning为True or False
        if not self.fine_tuning:
            if self.split:#根据split的值来处理音频数据和生成Mel频谱
                if audio.size(1) >= self.segment_size:#从音频数据中随机选择一个段落，并根据segment_size来截取相应长度的音频段
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:#用torch.nn.functional.pad函数在右侧进行填充，再使用mel_spectrogram函数生成Mel频谱。
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

                mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                      self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                      center=False)
            else: # validation step
                # match audio length to self.hop_size * n for evaluation
                if (audio.size(1) % self.hop_size) != 0:#如果音频长度不能被self.hop_size整除，则截取掉多余的成分
                    audio = audio[:, :-(audio.size(1) % self.hop_size)]
                mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                      self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                      center=False)#使用mel_spectrogram函数生成Mel频谱
                assert audio.shape[1] == mel.shape[2] * self.hop_size, "audio shape {} mel shape {}".format(audio.shape, mel.shape)
                #断言音频长度是否等于Mel频谱长度乘以self.hop_size
        else:#加载预先生成的Mel频谱文件，根据音频文件名构建相应的Mel频谱文件路径，并使用np.load加载文件
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:#将加载的mel频谱转换为张量，并根据需要添加维度。
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):#返回数据集中的样本数量。
        return len(self.audio_files)
