import torch
import torch.utils.data as data
import decord
import os
import numpy as np
from numpy.random import randint
import io
import pandas as pd
import random
from PIL import Image
import math
import copy
from decord import DECORDError   

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[-1])


class Video_dataset(data.Dataset):
    def __init__(self, root_path, list_file, labels_file,
                 num_segments=1, num_test_segments=16, modality='RGB', new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, val_mode=False,
                 index_bias=1, dense_sample=False, test_clips=3,
                 num_sample=1):

        self.root_path = root_path
        
        self.list_file = list_file
        self.labels_file = labels_file
        self.num_segments = num_segments
        self.num_test_segments = num_test_segments
        self.modality = modality
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.val_mode = val_mode
        self.loop = False
        self.index_bias = index_bias
        self.sample_range = 128
        self.dense_sample = dense_sample
        self.test_clips = test_clips
        self.num_sample = num_sample

        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.num_sample > 1:
            print('=> Using repeated augmentation...')

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()

    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if len(tmp[0]) == 3:
            if not self.test_mode:
                tmp = [item for item in tmp if int(item[1]) >= 8]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(type(self.video_list[0].path), self.video_list[0].path)
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, video_list):
        if self.dense_sample:
            sample_pos = max(1, 1 + len(video_list) - self.sample_range)
            interval = self.sample_range // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            base_offsets = np.arange(self.num_segments) * interval
            offsets = (base_offsets + start_idx) % len(video_list)
            return np.array(offsets) + self.index_bias
        else:
            seg_size = float(len(video_list) - 1) / self.num_segments
            offsets = []
            for i in range(self.num_segments):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                offsets.append(random.randint(start, end))
            return np.array(offsets) + self.index_bias

    def _get_val_indices(self, video_list):
        if self.dense_sample:
            sample_pos = max(1, 1 + len(video_list) - self.sample_range)
            t_stride = self.sample_range // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % len(video_list) for idx in range(self.num_segments)]
            return np.array(offsets) + self.index_bias
        else:
            tick = len(video_list) / float(self.num_segments)
            offsets = [int(tick * x) % len(video_list) for x in range(self.num_segments)]
            return np.array(offsets) + self.index_bias

    def _get_all_indices(self, video_list):
        indices = list(range(len(video_list)))
        return np.array(indices) + self.index_bias

    def _get_test_indices(self, video_list):
        if self.dense_sample:
            num_clips = self.test_clips
            sample_pos = max(0, len(video_list) - self.sample_range)
            interval = self.sample_range // self.num_segments
            start_list = [clip_idx * math.floor(sample_pos / (num_clips - 1)) for clip_idx in range(num_clips)]
            base_offsets = np.arange(self.num_segments) * interval
            offsets = []
            for start_idx in start_list:
                offsets.extend((base_offsets + start_idx) % len(video_list))
            return np.array(offsets) + self.index_bias
        else:
            num_clips = self.test_clips
            tick = len(video_list) / float(self.num_segments)
            start_list = np.linspace(0, tick - 1, num=num_clips, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [
                    int(start_idx + tick * x) % len(video_list)
                    for x in range(self.num_segments)
                ]
            return np.array(offsets) + self.index_bias

    def _decord_decode(self, video_path):
        try:
            container = decord.VideoReader(video_path)
        except Exception as e:
            print(f"Failed to decode {video_path} with exception: {e}")
            return None
        return container

    def __getitem__(self, index):
        if self.modality == 'video':
            _num_retries = 10
            for _ in range(_num_retries):
                record = copy.deepcopy(self.video_list[index])
                record_path = record.path
                # 防止 record.path 不是字符串
                if not isinstance(record_path, (str, bytes, os.PathLike)):
                    record_path = str(record_path)
                directory = os.path.join(self.root_path, record_path)
                # directory = os.path.join(self.root_path, record.path)
                video_list = self._decord_decode(directory)
                if video_list is None:
                    print(f"Failed to decode video idx {index} from {directory}")
                    index = random.randint(0, len(self.video_list) - 1)
                    continue
                break
        else:
            record = self.video_list[index]
            video_list = os.listdir(os.path.join(self.root_path, record.path))

        if self.test_mode:
            segment_indices = self._get_test_indices(video_list)
        elif self.val_mode:
            raise NotImplementedError("val_mode currently not supported in this version.")
        else:  # train mode
            segment_indices = self._sample_indices(video_list) if self.random_shift else self._get_val_indices(video_list)

        return self.get(record, video_list, segment_indices)

    # ✅ 改造后的 get 方法
    def get(self, record, video_list, indices):
        images = []
        total_frames = len(video_list)

        for seg_ind in indices:
            p = int(seg_ind) - 1  # decord VideoReader 索引从 0 开始
            if p < 0:
                p = 0
            elif p >= total_frames:
                p = total_frames - 1  # 超界则用最后一帧

            try:
                frame = video_list[p].asnumpy()
                seg_imgs = [Image.fromarray(frame).convert('RGB')]
            except DECORDError:
                # print(f"[WARN] Decode error: {record.path}, frame {p}/{total_frames}, fallback to previous frame")
                try:
                    # 如果不是第一帧，就用上一帧；否则用下一帧
                    if p > 0:
                        fallback_idx = p - 1
                    else:
                        fallback_idx = 1 if total_frames > 1 else 0

                    seg_imgs = [Image.fromarray(video_list[fallback_idx].asnumpy()).convert('RGB')]
                except Exception:
                    new_index = random.randint(0, len(self.video_list) - 1)
                    return self.__getitem__(new_index)


            images.extend(seg_imgs)

        process_data, record_label = self.transform((images, record.label))
        return process_data, record_label

    def __len__(self):
        return len(self.video_list)

class Video_dataset_few(data.Dataset):
    def __init__(self, root_path, list_file, labels_file,
                 num_segments=1, num_test_segments=16, modality='RGB', new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, val_mode=False,
                 index_bias=1, dense_sample=False, test_clips=3,
                 num_sample=1, num_shots=2):  # 添加 num_shots 参数

        self.root_path = root_path
        self.list_file = list_file
        self.labels_file = labels_file
        self.num_segments = num_segments
        self.num_test_segments = num_test_segments
        self.modality = modality
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.val_mode = val_mode
        self.loop = False
        self.index_bias = index_bias
        self.sample_range = 128
        self.dense_sample = dense_sample
        self.test_clips = test_clips
        self.num_sample = num_sample
        self.num_shots = num_shots  # 用于控制每类样本数

        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.num_sample > 1:
            print('=> Using repeated augmentation...')
        print(f'Using {num_shots}-shot strategy')
        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()

    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    # 添加一个方法来获取每个类别的2个随机样本
    def _get_random_shot_samples(self, class_id):
        # 获取每个类别的所有样本
        class_samples = [item for item in self.video_list if item.label == class_id]
        # 随机选择2个样本
        selected_samples = random.sample(class_samples, self.num_shots)
        return selected_samples

    # 修改 __getitem__ 方法来使用 2-shot 随机采样
    def __getitem__(self, index):
        # 获取视频记录
        record = self.video_list[index]
        video_list = os.listdir(os.path.join(self.root_path, record.path))

        # 使用随机2-shot采样（在训练模式下）
        if not self.test_mode:
            # 获取记录的视频的类别 ID
            class_id = record.label
            # 获取该类别的2个随机样本
            random_samples = self._get_random_shot_samples(class_id)

            # 选择一个样本（这里选取第一个样本，实际使用中可以改成其他策略）
            selected_record = random_samples[0]
            selected_video_list = os.listdir(os.path.join(self.root_path, selected_record.path))
            segment_indices = self._sample_indices(selected_video_list)
        else:
            # 如果是测试模式，则使用常规的采样方式
            segment_indices = self._get_test_indices(video_list)

        return self.get(record, video_list, segment_indices)

    # 其余部分可以保持不变，确保视频采样部分与数据加载过程一致

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if len(tmp[0]) == 3:
            if not self.test_mode:
                tmp = [item for item in tmp if int(item[1]) >= 8]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(f'Video number: {len(self.video_list)}')

    def _sample_indices(self, video_list):
        if self.dense_sample:
            sample_pos = max(1, 1 + len(video_list) - self.sample_range)
            interval = self.sample_range // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            base_offsets = np.arange(self.num_segments) * interval
            offsets = (base_offsets + start_idx) % len(video_list)
            return np.array(offsets) + self.index_bias
        else:
            seg_size = float(len(video_list) - 1) / self.num_segments
            offsets = []
            for i in range(self.num_segments):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                offsets.append(random.randint(start, end))
            return np.array(offsets) + self.index_bias

    def get(self, record, video_list, indices):
        images = []
        total_frames = len(video_list)

        for seg_ind in indices:
            p = int(seg_ind) - 1  # decord VideoReader 索引从 0 开始
            if p < 0:
                p = 0
            elif p >= total_frames:
                p = total_frames - 1  # 超界则用最后一帧

            try:
                frame = video_list[p].asnumpy()
                seg_imgs = [Image.fromarray(frame).convert('RGB')]
            except DECORDError:
                # 如果不是第一帧，就用上一帧；否则用下一帧
                try:
                    if p > 0:
                        fallback_idx = p - 1
                    else:
                        fallback_idx = 1 if total_frames > 1 else 0
                    seg_imgs = [Image.fromarray(video_list[fallback_idx].asnumpy()).convert('RGB')]
                except Exception:
                    new_index = random.randint(0, len(self.video_list) - 1)
                    return self.__getitem__(new_index)

            images.extend(seg_imgs)

        process_data, record_label = self.transform((images, record.label))
        return process_data, record_label

    def __len__(self):
        return len(self.video_list)