import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224), # 裁剪
                transforms.RandomHorizontalFlip(), # 以给定的概率随机水平翻转给定的图像
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), # 使用均值和标准差对张量图像进行归一化
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch = zip(*data)

        image_batch = torch.stack(image_batch, 0) # 16len ([2, 3, 224, 224], ..., [2, 3, 224, 224])
        max_seq_length = max(seq_lengths_batch)

        target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int) # (batch_size, seq_len)
        target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int) 
        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids
        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks
        # print(len(target_batch[0]))
        # print()
        # print(len(target_tf_idf[0]))
        # return image_id_batch, image_batch, torch.LongTensor(target_batch), torch.FloatTensor(target_masks_batch), torch.FloatTensor(target_tf_idf)
        return image_id_batch, image_batch, torch.LongTensor(target_batch), torch.FloatTensor(target_masks_batch)
        # tuple: len=batch_size "CXR2384_IM-0942"等, [batch_size, image_num, 3, 224, 224]
        # [batch_size, max_seq_len], [batch_size, max_seq_len]