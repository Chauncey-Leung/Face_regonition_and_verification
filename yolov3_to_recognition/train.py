import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from yolov3_to_recognition.nets.yolo import YoloBody
from yolov3_to_recognition.nets.yolo_training import YOLOLoss
from yolov3_to_recognition.utils.callbacks import LossHistory
from yolov3_to_recognition.utils.dataloader import YoloDataset, yolo_dataset_collate
from yolov3_to_recognition.utils.utils_fit import fit_one_epoch, get_classes, get_anchors


if __name__ == "__main__":
    Cuda = True  # 是否使用Cuda
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    model_path = 'logs/ep050-loss4.062-val_loss3.919.pth'
    classes_path = 'model_data/my_class.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    input_shape = [416, 416]  # 输入的shape大小，一定要是32的倍数

    # pretrained = False

    # 训练分为两个阶段，分别是冻结阶段和解冻阶段

    Init_Epoch = 1  # 从第几轮开始训练
    Freeze_Epoch = 50  # 0~Freeze_Epoch为冻结阶段训练
    UnFreeze_Epoch = 100  # Freeze_Epoch+1~UnFreeze_Epoch为解冻阶段训练
    # 冻结阶段训练参数，此时模型的主干被冻结了，特征提取网络不发生改变，占用的显存较小，仅对网络进行微调
    Freeze_batch_size = 8
    Freeze_lr = 1e-3
    # 解冻阶段训练参数, 此时模型的主干不被冻结了，特征提取网络会发生改变,占用的显存较大，网络所有的参数都会发生改变
    Unfreeze_batch_size = 2
    Unfreeze_lr = 1e-4
    # 是否进行冻结训练，默认先冻结主干训练后解冻训练。
    Freeze_Train = True
    # 线程数
    num_workers = 4
    # 获得图片路径和标签
    train_annotation_path = 'train.txt'
    val_annotation_path = 'val.txt'
    # 获取classes和anchor
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)
    # 创建yolo模型
    model = YoloBody(anchors_mask, num_classes, pretrained=False)

    print('Load weights {}.'.format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)
    loss_history = LossHistory("logs/")

    # 读取数据集对应的txt
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()

