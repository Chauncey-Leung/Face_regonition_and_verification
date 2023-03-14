import colorsys
import os
import time

import numpy as np
import torch
from torch import nn
from PIL import ImageDraw, ImageFont, Image
from yolov3_to_recognition.nets.yolo import YoloBody
from yolov3_to_recognition.utils.utils_bbox import DecodeBox
from yolov3_to_recognition.utils.utils_fit import get_anchors, get_classes


class YOLO(object):
    _defaults = {
        # 这里之前用的是相对路径，但是会报错找不到路径，因此改为绝对路径，在新的环境中需要修改路径，根据代码报错提示修改即可
        # 指向\yolov3_to_recognition\logs下权值文件
        "model_path": r'F:\Present_Tasks\Photoelectric_Image_Processing\face_regonition_and_verification\yolov3_to_recognition\logs\ep100-loss3.350-val_loss3.394.pth',
        # 指向\yolov3_to_recognition\model_data\my_class.txt
        "classes_path": r'F:\Present_Tasks\Photoelectric_Image_Processing\face_regonition_and_verification\yolov3_to_recognition\model_data\my_class.txt',
        # 指向\yolov3_to_recognition\model_data\yolo_anchors.txt
        "anchors_path": r'F:\Present_Tasks\Photoelectric_Image_Processing\face_regonition_and_verification\yolov3_to_recognition\model_data\yolo_anchors.txt',

        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],  # anchors_mask用于帮助代码找到对应的先验框
        "input_shape": [416, 416],  # 输入图片的大小，必须为32的倍数
        "confidence": 0.5,  # 只有得分大于置信度的预测框会被保留下来
        "nms_iou": 0.3,  # 非极大抑制所用到的nms_iou大小
        "letterbox_image": False,  # 该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize
        "cuda": True,  # 是否使用Cuda
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        #  获得种类和先验框的数量
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)

        # 画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    #   生成模型
    def generate(self):
        # 建立yolov3模型，载入yolov3模型的权重
        self.net = YoloBody(self.anchors_mask, self.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # 检测图片
    def detect_image(self, image):
        bbox_location = []
        image_shape = np.array(np.shape(image)[0:2])
        # 这里将图像转换成RGB格式，因为有预训练的权重，网络效果好，而且方便绘图
        image = cvtColor(image)
        # 给图像增加灰条，实现不失真的resize
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # 添加batch_size维度, /255.0是为了归一化，喂给网络
        image_data = np.expand_dims(np.transpose(np.array(image_data, dtype='float32') / 255.0, (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # 以上操作，将我们的输入图像转换成网络需要的输入数据，接下来，便是将其输给网络，得到输出
            outputs = self.net(images)
            # 输出的数据需要进行解码，才能转换成我们能看懂的、可视化的结果。decode_box()即边界框解码函数
            outputs = self.bbox_util.decode_box(outputs)
            # 将预测框进行堆叠，然后进行非极大抑制
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1),
                                                         self.num_classes,
                                                         self.input_shape,
                                                         image_shape,
                                                         self.letterbox_image,
                                                         conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
            if results[0] is None:
                return image
            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]  # 置信度
            top_boxes = results[0][:, :4]
        # 设置字体与边框厚度
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1)) // 2

        # 图像绘制
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            draw = ImageDraw.Draw(image)

            print(predicted_class, score, top, left, bottom, right)
            item_tuple = predicted_class, top, left, bottom, right
            bbox_location.append(item_tuple)

            # 用来显示置信度用的代码，因为人数众多显示的置信度会遮挡其他人脸，为了展示效果故将此段代码注释
            # label = '{} {:.2f}'.format(predicted_class, score)
            # label_size = draw.textsize(label, font)
            # label = label.encode('utf-8')
            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            # else:
            #     text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            # draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)

        # 图片左上角标出总人数
        total_num_label = 'Total num of faces: {:d}'.format(len(bbox_location))
        total_num_label = total_num_label.encode('utf-8')
        draw.text((100, 100), str(total_num_label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

        return image, bbox_location


def cvtColor(image):
    """代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB"""
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size, letterbox_image):
    """对输入图像进行resize"""
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image
