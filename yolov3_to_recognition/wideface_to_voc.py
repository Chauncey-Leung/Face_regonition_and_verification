import os
import random
import shutil
from skimage import io

'''
    改文件用于将wideface数据集转换为voc数据集
'''

# voc数据集格式
# 文件头
headstr = """\
<annotation>
    <folder>VOC2007</folder>
    <filename>%06d.jpg</filename>
    <source>
        <database>My Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>?</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>NULL</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
# 有多少个目标就有多少个objstr，有多个目标则往list中append即可
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
# 文件还需要把</annotation>标签对的后半加在文件结尾
tailstr = '''\
</annotation>
'''


def all_path(filename):
    root_path = r'F:\Present_Tasks\Photoelectric_Image_Processing\face_regonition_and_verification' \
                r'\yolov3_to_recognition\VOC2007_face'
    # root_path = r'F:\Neural_Network\my_yolov3\VOC2007_face'
    return os.path.join(root_path, filename)


def writexml(idx, head, bbxes, tail):
    filename = all_path("Annotations/%06d.xml" % idx)
    f = open(filename, "w")
    f.write(head)
    for bbx in bbxes:
        f.write(objstr % ('face', bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]))
    f.write(tail)
    f.close()

# 清空voc路径下遗留的文件并建立文件路径，以保证数据集格式的准确
def clear_dir():
    if shutil.os.path.exists(all_path('Annotations')):
        shutil.rmtree(all_path('Annotations'))
    if shutil.os.path.exists(all_path('ImageSets')):
        shutil.rmtree(all_path('ImageSets'))
    if shutil.os.path.exists(all_path('JPEGImages')):
        shutil.rmtree(all_path('JPEGImages'))

    shutil.os.mkdir(all_path('Annotations'))
    shutil.os.makedirs(all_path('ImageSets/Main'))
    shutil.os.mkdir(all_path('JPEGImages'))


def excute_datasets(idx, datatype):
    f = open(all_path('ImageSets/Main/' + datatype + '.txt'), 'a')
    f_bbx = open(all_path('wider_face_split/wider_face_' + datatype + '_bbx_gt.txt'), 'r')

    while True:
        # 读入图片名称filename
        '''
            该txt文件的格式如下:
            filename
            Number of bounding box
            x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
        '''
        filename = f_bbx.readline().strip('\n')
        if not filename:
            break
        # 读入filename对应的图片
        im = io.imread(all_path('WIDER_' + datatype + '/images/' + filename))
        # 填充xml文件的head中的<filename> <height> <width> <deepth>
        head = headstr % (idx, im.shape[1], im.shape[0], im.shape[2])
        # 读入该图片含有bounding box数
        nums = f_bbx.readline().strip('\n')
        bbxes = []
        # 如果该照片bounding box数为0，则执行特殊操作，不能和有目标时候的情况一致处理
        if nums == '0':
            bbx_info = f_bbx.readline()
            continue
        #  遍历各bounding box
        for ind in range(int(nums)):
            bbx_info = f_bbx.readline().strip(' \n').split(' ')
            bbx = [int(bbx_info[i]) for i in range(len(bbx_info))]
            # x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
            if bbx[7] == 0:
                bbxes.append(bbx)
        writexml(idx, head, bbxes, tailstr)
        # 把widerface数据集下的jpeg文件复制到voc数据集对应的文件路径下
        shutil.copyfile(all_path('WIDER_' + datatype + '/images/' + filename), all_path('JPEGImages/%06d.jpg' % idx))
        f.write('%06d\n' % idx)
        idx += 1
    f.close()
    f_bbx.close()
    return idx

# 打乱样本
def shuffle_file(filename):
    f = open(filename, 'r+')
    lines = f.readlines()
    random.shuffle(lines)
    f.seek(0)
    f.truncate()
    f.writelines(lines)
    f.close()


if __name__ == '__main__':
    clear_dir()
    idx = 1
    idx = excute_datasets(idx, 'train')
    idx = excute_datasets(idx, 'val')
    print('Complete!')
    