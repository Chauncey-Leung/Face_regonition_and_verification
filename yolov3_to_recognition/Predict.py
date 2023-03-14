import os.path
from PIL import Image
from yolov3_to_recognition.yolo import YOLO


#  封装成函数方便face_recognition_and_verification.py文件直接调用
# face_recognition_and_verification.py文件实现的是识别与验证流程化实现
def face_to_recognition(model, dir_origin_path, dir_save_path, isCrop=True, isSave=True):
    # 请将要进行人脸识别的图片存在yolov3_to_recognition/img文件夹下
    img_name = input('Input image filename:')
    img = os.path.join(dir_origin_path, img_name)
    try:
        image = Image.open(img)
        image_orign = image.copy()
    except:
        print('Open Error! Try again!')
    else:
        # 平时不用修改。每次改路径img[x:]里面的x都要修改
        save_path = os.path.join(dir_save_path, img[26:])
        r_image, r_bbox_location = model.detect_image(image)
        face_num = 0
        if isCrop:
            # 创建文件夹去保存截取的人脸照片，文件位于img_out中的子文件夹
            # 截取下来的图片用于后续人脸验证
            isExists = os.path.exists(dir_save_path + img_name.split(".")[0])
            if not isExists:
                os.makedirs(dir_save_path + img_name.split(".")[0])
            print("Directory created successfull\n")
            for sub_bbox in r_bbox_location:
                if sub_bbox[0] == 'face':
                    face_num += 1
                    sub_img = image_orign.crop((sub_bbox[2], sub_bbox[1], sub_bbox[4], sub_bbox[3]))
                    sub_img.save(dir_save_path + img_name.split(".")[0] + '/' + str(face_num) + '.jpeg')
        if isSave:
            # 框出识别到的人脸的图片保存在了img_out文件夹下，图片名称与输入图片相同
            r_image.save(save_path)
        # 返回bouding box在图像中的位置 人脸数量 文件名称 原始图像 标注出人脸的图像
        return r_bbox_location, face_num, img_name.split(".")[0], image_orign, r_image


# 如果只是目标识别，只要运行该文件即可
if __name__ == "__main__":
    yolo = YOLO()
    dir_origin_path = "img/"  # 指定了用于检测的图片的文件夹路径
    dir_save_path = "img_out/"  # 指定了检测图片的保存路径
    isCrop = True  # 是否截取人脸照片并创建文件夹保存，选true用于后续人脸验证
    isSave = True  # 是否保存的图片
    # isNumInConsole = True  # 是否命令行显示图片中人脸数量

    while True:
        # 请将要进行人脸识别的图片存在yolov3_to_recognition/img文件夹下
        img_name = input('Input image filename:')
        img = os.path.join(dir_origin_path, img_name)
        try:
            image = Image.open(img)
            # 有时因为输入图像格式问题，需要加这一行解决error
            # image = image.convert('RGB')

            # 手动备份原始图像，因为后面的操作都不会拷贝新的内存空间备份原图像
            image_orign = image.copy()
        except:
            print('Open Error! Try again!')
            continue
        else:
            save_path = os.path.join("img_out", img[4:])
            r_image, r_bbox_location = yolo.detect_image(image)
            face_num = 0
            # if isNumInConsole:
            #     print("Total num of faces: " % len(r_bbox_location))

            if isCrop:
                # 创建文件夹去保存截取的人脸照片，文件位于img_out中的子文件夹
                isExists = os.path.exists(dir_save_path + img_name.split(".")[0])
                if not isExists:
                    os.makedirs(dir_save_path + img_name.split(".")[0])
                print("Directory created successfull\n")
                for sub_bbox in r_bbox_location:
                    if sub_bbox[0] == 'face':
                        face_num += 1
                        sub_img = image_orign.crop((sub_bbox[2], sub_bbox[1], sub_bbox[4], sub_bbox[3]))
                        sub_img.save(dir_save_path + img_name.split(".")[0] + '/' + str(face_num) + '.jpeg')
            r_image.show()
            if isSave:
                # 框出识别到的人脸的图片保存在了img_out文件夹下，图片名称与输入图片相同
                r_image.save(save_path)
