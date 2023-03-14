import os
import random
import xml.etree.ElementTree as ET
from yolov3_to_recognition.utils.utils_fit import get_classes

'''
    根据voc数据集的annotation，生成方便网络处理数据的tain.txt 和 val.txt
    更改文件路径后需要重新运行该文件，更新tain.txt 和 val.txt
    需要注意tain.txt 和 val.txt 绝对路径不能含有中文（识别不了）以及带有空格的文件夹名称（字符串会从空格处截断）
'''

classes_path = 'yolov3_to_recognition/model_data/my_class.txt'
# 指向VOC数据集所在的文件夹
VOCdevkit_path = 'VOC2007_face'
VOCdevkit_sets = ['train', 'val']
classes, _ = get_classes(classes_path)


def convert_annotation(image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'Annotations/%s.xml' % image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        # 对应<difficult>，0表述物体容易识别，1表示物体难识别
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        # 对应<name>，不同类别
        cls = obj.find('name').text
        # 判断类别是否包含在my_classes中，不包含或者难以识别即忽略该目标
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        # 对应<bndbox>，bounding box的四个位置参数
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)
    print("Generate train.txt and val.txt for train.")
    # 循环两次，分别处理训练集和验证集
    for image_set in VOCdevkit_sets:
        # image_ids是训练集或验证集对应的文件的名称
        image_ids = open(os.path.join(VOCdevkit_path, 'ImageSets/Main/%s.txt' % image_set),
                         encoding='utf-8').read().strip().split()
        # 新建tain.txt 或 val.txt
        list_file = open('%s.txt' % image_set, 'w', encoding='utf-8')
        # 把转换后的数据逐行存入新建的txt
        for image_id in image_ids:
            list_file.write('%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), image_id))
            convert_annotation(image_id, list_file)
            list_file.write('\n')
        list_file.close()
    print("Generate train.txt and val.txt for train done.")
