import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from siamese_to_verification.nets.siamese import Siamese
from siamese_to_verification.utils.dataloader import SiameseDataset, dataset_collate
from siamese_to_verification.utils.utils_fit import fit_one_epoch


#  计算图片总数
def get_image_num(path):
    num = 0
    train_path = os.path.join(path, 'face_verification')
    for character in os.listdir(train_path):
        #   在大种类下遍历小种类
        character_path = os.path.join(train_path, character)
        num += len(os.listdir(character_path))
    return num


if __name__ == "__main__":
    Cuda = True
    # 数据集存放的路径
    dataset_path = "./datasets"
    # 输入图像的大小，默认为105,105,3
    input_shape = [105, 105, 3]
    pretrained = True
    model_path = ""

    model = Siamese(input_shape, pretrained)
    if model_path != '':
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 这里是浅拷贝，正好
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        # 需要修改预训练参数使其与Siamese适配
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
        # 如果网络的输入数据维度或类型上变化不大，设置torch.backends.cudnn.benchmark = true可以增加运行效率
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss = nn.BCELoss()
    #   训练集和验证集的比例。
    train_ratio = 0.7
    images_num = get_image_num(dataset_path)
    num_train = int(images_num * train_ratio)
    num_val = images_num - num_train

    if True:
        Batch_size = 4
        # Batch_size = 1  # 单纯为了debugger用
        Lr = 1e-4
        Init_epoch = 0
        Freeze_epoch = 50
        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        optimizer = optim.Adam(model_train.parameters(), Lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
        train_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=True)
        val_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=dataset_collate)

        for epoch in range(Init_epoch, Freeze_epoch):
            fit_one_epoch(model_train, model, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          Freeze_epoch, Cuda)
            lr_scheduler.step()

    if True:
        Batch_size = 16
        Lr = 1e-5
        Freeze_epoch = 50
        Unfreeze_epoch = 100
        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集!')

        optimizer = optim.Adam(model_train.parameters(), Lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=True)
        val_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=dataset_collate)

        for epoch in range(Freeze_epoch, Unfreeze_epoch):
            fit_one_epoch(model_train, model, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          Unfreeze_epoch, Cuda)
            lr_scheduler.step()
