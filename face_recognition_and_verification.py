import os
import numpy as np
from PIL import ImageDraw, Image
from yolov3_to_recognition.Predict import face_to_recognition
from yolov3_to_recognition.yolo import YOLO
from siamese_to_verification.predict import face_to_verification
from siamese_to_verification.siamese import Siamese
import matplotlib.pyplot as plt



yolo = YOLO()
siamese = Siamese()
dir_origin_path = "yolov3_to_recognition/img/"  # 指定了用于检测的图片的文件夹路径
dir_save_path = "yolov3_to_recognition/img_out/"  # 指定了检测图片的保存路径
isCrop = True  # face recognition时是否截取人脸照片并创建文件夹保存，选true用于后续faceverification
isSave = True  # 是否保存的图片(用红框框标出人脸，左上角有总人数的那种)

# image_1是我本人的照片（参考照片，比如证件照)
image1_path = "siamese_to_verification/fixed_lqx.jpeg"

image2_dir_raw = "yolov3_to_recognition/img_out"
dir_save_path_verification = 'siamese_to_verification/img_out'


if __name__ == "__main__":
    r_bbox_location, face_num, img_name_without_suffix, image_orign, image_recognition = \
        face_to_recognition(yolo, dir_origin_path, dir_save_path, isCrop=True, isSave=True)

    plt.figure(figsize=(15.0, 15.0), dpi=72)
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(image_recognition))
    plt.axis('off')

    # 将image_1与image_2_dir中的文件一一比较，结果存在probability中
    probability = []
    image2_dir = os.path.join(image2_dir_raw, img_name_without_suffix)

    probability = face_to_verification(siamese, image1_path, image2_dir)

    for i in range(len(r_bbox_location)):
        print('%d, Similarity:%.3f' % (i + 1, probability[i]))
    # 取image_2_dirt中相似度最高的图片作为face verification的结果
    max_probability = max(probability)
    max_probability_index = probability.index(max_probability)
    # 将其在原图中标注出来
    draw = ImageDraw.Draw(image_orign)
    draw = ImageDraw.Draw(image_orign)
    thickness = int(max((image_orign.size[0] + image_orign.size[1]) // np.mean(image_orign.size), 1)) * 2
    for i in range(thickness):
        # draw.rectangle([x0 + i, y0 + i, x1 - i, y1 - i], outline='red')
        draw.rectangle([r_bbox_location[max_probability_index][2] + i,
                        r_bbox_location[max_probability_index][1] + i,
                        r_bbox_location[max_probability_index][4] - i,
                        r_bbox_location[max_probability_index][3] - i], outline='red')
    del draw
    image_orign.save(os.path.join(dir_save_path_verification, img_name_without_suffix + '.jpeg'))
    image_verification = Image.open(os.path.join(dir_save_path_verification, img_name_without_suffix + '.jpeg'))
    plt.subplot(1, 2, 2)
    plt.imshow(np.array(image_verification))
    plt.axis('off')
    plt.text(-10, -10, 'Totol num:%d' % len(probability), ha='center', va='bottom', fontsize=11)
    plt.show()
