import os

from PIL import Image
from siamese_to_verification.siamese import Siamese


def face_to_verification(model, image1_path, image2_dir, probability=[]):
    try:
        image_1 = Image.open(image1_path)
    except:
        print('Image_1 Open Error! Try again!')
    image2_list = os.listdir(image2_dir)

    # !!!这段代码一定要注意，查了很久才发现错误
    # list_str的排序是1.jpeg 10.jpeg 2.jpeg……
    # 需要转换成1.jpeg  2.jpeg …… 10.jpeg
    temp = []
    for i in range(len(image2_list)):
        temp.append(image2_list[i][0:-5])
    temp.sort(key=int)
    image2_list = []
    for j in range(len(temp)):
        image2_list.append(temp[j] + '.jpeg')

    for image_2 in image2_list:
        try:
            image_2 = Image.open(os.path.join(image2_dir, image_2))
        except:
            print('Image_2 Open Error! Try again!')
        probability_single = model.compare_image(image_1, image_2)
        probability.append(probability_single)
    return probability


if __name__ == "__main__":
    model = Siamese()
    probability = []

    while True:
        # image_1是我本人的照片（参考照片，比如证件照）。该照片之前作为数据集参与了网络的训练与验证，现在直接引用其在数据集下的路径就好
        image_1 = 'siamese_to_verificaton/datasets/face_verification/lqx/007.jpeg'
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue
        image_2_dir = input('Input image_2 filename:')  # 1,2,3
        # 然后将image_1与image_2_filelist中的文件一一比较，取image_2_filelist中相似度最高的图片作为识别结果
        image_2_filepath = os.path.join('F:\\Neural_Network\\my_yolov3\\img_out', image_2_dir)
        image_2_list = os.listdir(image_2_filepath)
        for image_2 in image_2_list:
            try:
                image_2 = Image.open(os.path.join(image_2_filepath, image_2))
            except:
                print('Image_2 Open Error! Try again!')
                continue
            probability_single = model.detect_image(image_1, image_2)
            probability.append(probability_single)
        num = 1
        for i in probability:
            print('%d, Similarity:%.3f' % (num, i))
            num += 1
