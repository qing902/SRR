import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np

class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels
        # Load samples
        x_data = dataset["samples"]

        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif (len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels):
            x_data = x_data.transpose(0, 2, 1)

        # Convert to torch tensor·
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        # Load labels
        y_data = dataset.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)

        # Normalize data
        if dataset_configs.normalize:
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)

        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        return x, y, index

    def __len__(self):
        return self.len

def data_generator(data_path, domain_id, dataset_configs, hparams, dtype):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))

    # Loading datasets
    dataset = Load_Dataset(dataset_file, dataset_configs)

    if dtype == "test":  # you don't need to shuffle or drop last batch while testing
        shuffle = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last

    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=0)

    return data_loader


# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# import numpy as np
# import pywt   # 用于小波分解
# import random
#
#
# class Load_Dataset(Dataset):
#     def __init__(self, dataset, dataset_configs, num_scales=3, wavelet="db4", mask_ratio=0.3):
#         super().__init__()
#         self.num_channels = dataset_configs.input_channels
#         self.num_scales = num_scales  # 分解层数 J
#         self.wavelet = wavelet
#         self.mask_ratio = mask_ratio
#
#         # Load samples
#         x_data = dataset["samples"]  # (N, C, L)
#
#         # 保证维度正确
#         if len(x_data.shape) == 2:
#             x_data = x_data.unsqueeze(1)  # (N,1,L)
#         elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
#             x_data = x_data.transpose(0, 2, 1)
#
#         if isinstance(x_data, np.ndarray):
#             x_data = torch.from_numpy(x_data)
#
#         # Load labels
#         y_data = dataset.get("labels")
#         if y_data is not None and isinstance(y_data, np.ndarray):
#             y_data = torch.from_numpy(y_data)
#
#         # Normalize
#         if dataset_configs.normalize:
#             data_mean = torch.mean(x_data, dim=(0, 2))
#             data_std = torch.std(x_data, dim=(0, 2))
#             self.transform = transforms.Normalize(mean=data_mean, std=data_std)
#         else:
#             self.transform = None
#
#         self.x_data = x_data.float()
#         self.y_data = y_data.long() if y_data is not None else None
#         self.len = x_data.shape[0]
#
#     def dwt_multiscale(self, signal):
#         """
#         对单条多通道信号做 DWT+IDWT
#         输入: (C, L)
#         输出: (num_scales+1, L, C)
#         """
#         C, L = signal.shape
#         comps = []
#
#         for c in range(C):
#             coeffs = pywt.wavedec(signal[c].numpy(), wavelet=self.wavelet, level=self.num_scales)
#             # coeffs = [cJ, dJ, ..., d1]
#
#             for i in range(len(coeffs)):
#                 coeffs_zero = [np.zeros_like(c) for c in coeffs]
#                 coeffs_zero[i] = coeffs[i]
#                 recon = pywt.waverec(coeffs_zero, wavelet=self.wavelet)[:L]
#                 comps.append(recon)
#
#         comps = np.array(comps).reshape(C, self.num_scales+1, L)
#         comps = torch.from_numpy(comps).permute(1, 2, 0)  # (num_scales+1, L, C)
#         return comps
#
#     def apply_mask(self, seq):
#         """
#         对序列随机mask
#         输入: (L, C)
#         输出: masked_seq
#         """
#         L, C = seq.shape
#         mask = torch.rand(L) > self.mask_ratio
#         masked_seq = seq.clone()
#         masked_seq[~mask] = 0.0
#         return masked_seq
#
#     def __getitem__(self, index):
#         x = self.x_data[index]  # (C, L)
#
#         if self.transform:
#             x = self.transform(x.reshape(self.num_channels, -1, 1)).reshape(x.shape)
#
#         x = x.permute(0, 1)  # (C, L)
#
#         # -------- 多尺度分解+重构 (4.2) --------
#         multi_scales = self.dwt_multiscale(x)  # (num_scales+1, L, C)
#
#         # -------- 随机mask (4.3) --------
#         masked_scales = torch.stack([self.apply_mask(seq) for seq in multi_scales])  # (num_scales+1, L, C)
#
#         y = self.y_data[index] if self.y_data is not None else None
#         return multi_scales.float(), masked_scales.float(), y, index
#
#     def __len__(self):
#         return self.len
