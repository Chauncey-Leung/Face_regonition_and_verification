# Face_regonition_and_verification

## regonition

输出图像中的人数，实质上是目标检测问题，目标类别仅“人脸”一个类别。一般目标检测网络进行目标检测的过程中，分两阶段进行：
①提取潜在的候选框；②用分类器逐一筛选每个候选框。暴力遍历的存在使得检测实时性差，同时两阶段需要单独训练，难以优化。
YOLO直接将整张图片作为网络输入，仅通过一次向前推断，即可在输出层同时得到边界框定位和相关的分类，从而速度更快；
YOLO直接分析全图，能更好地编码上下文信息，如物体与物体的关系，而不仅仅局限于物体的外观；
YOLO的泛化能力也很好，即便应用在与训练集分布不一致的情况下也能取得良好表现。
综上，我选择YOLO作为我的人脸识别方法

## verification

人脸验证问题比人脸识别更困难，因为要想要验证人脸，必须先识别出人脸。
起初我想沿用YOLOv3的结构，将人脸作为一类，秉承YOLOv2中提出的Hierarchical classification思想，将face类下设置专业同学数目一样多的子类，每个子类代表某一位同学的脸，即使用multi-label模型，进行训练，即最后输入我的照片，网络不仅知道照片中存在目标属于face类，还知道该目标属于face_of_LiangQianxi类。但这么做有一个难点，大三信息专业将近100人，我没有每个人的私人照，也不认识所有人，而且我担心不同人脸之间的特征差别未必明显。因此，如何从较小的数据中去学习，如何增添新的类别时不需要重新训练神经网络，是我需要克服的问题。Siamese Neural Network可满足要求。


## 程序的整体功能和构架概括

“face_recognition_and_verification.py”通过深度学习算法和目标检测技术实现了人脸识别和验证的功能；“siamese_to_verification.py”实现了Siamese神经网络用于人脸验证的训练和预测功能；“yolov3_to_recognition”则是一个用于目标检测和人脸识别的深度学习算法。

下面是每个文件的具体功能表格：

| 文件名 | 功能 |
| :-----| :---- |
| face_recognition_and_verification.py | 实现了人脸识别及人脸验证的功能 |
| siamese_to_verification\predict.py | 实现了Siamese神经网络的预测功能 |
| siamese_to_verification\siamese.py | 提供了Siamese神经网络模型的实现，包括输入数据预处理和相似度比较等 |
| siamese_to_verification\train.py | 实现了Siamese神经网络的训练和验证功能 |
| siamese_to_verification\nets\siamese.py | 实现了Siamese神经网络模型 |
| siamese_to_verification\utils\dataloader.py | 提供了用于Siamese神经网络训练和验证的数据集加载和预处理模块，提供数据处理函数 |
| siamese_to_verification\utils\utils_fit.py | 实现了深度学习模型的训练和验证的逻辑 |
| yolov3_to_recognition\Predict.py | 实现了目标检测和人脸识别的功能 |
| yolov3_to_recognition\train.py | 实现了YOLOv3模型的训练功能 |
| yolov3_to_recognition\voc_annotation.py | 实现了将数据格式转换为目标检测算法需要的数据集格式 |
| yolov3_to_recognition\wideface_to_voc.py | 实现了将一个数据集转换为目标检测算法需要的数据集格式 |
| yolov3_to_recognition\yolo.py | 提供了YOLOv3对象检测算法的具体实现 |
| yolov3_to_recognition\nets\yolo.py | 实现了YOLOv3对象检测算法 |
| yolov3_to_recognition\nets\yolo_training.py | 实现了YOLOv3对象检测算法的损失函数 |
| yolov3_to_recognition\utils\callbacks.py | 实现了记录训练过程中损失值和绘制损失曲线图的功能 |
| yolov3_to_recognition\utils\dataloader.py | 用于加载数据集 |
| face_recognition_and_verification.py | 程序的主入口，集成了人脸检测、识别和验证 |
| siamese_to_verification/predict.py | siamese模型的预测函数 |
| siamese_to_verification/siamese.py | siamese模型的实现 |
| siamese_to_verification/train.py | siamese模型的训练函数 |
| siamese_to_verification/nets/siamese.py | siamese模型网络结构的定义 |
| siamese_to_verification/utils/dataloader.py | siamese模型数据加载器的实现 |
| siamese_to_verification/utils/utils_fit.py | siamese模型训练的工具函数 |
| yolov3_to_recognition/Predict.py | yolo模型的预测函数 |
| yolov3_to_recognition/train.py | yolo模型的训练函数 |
| yolov3_to_recognition/voc_annotation.py | 将标注数据转换为训练用的格式 |
| yolov3_to_recognition/wideface_to_voc.py | 将宽面数据转换为训练用的格式 |
| yolov3_to_recognition/yolo.py | yolo模型实现 |
| yolov3_to_recognition/nets/yolo.py | yolo模型网络结构的定义 |
| yolov3_to_recognition/nets/yolo_training.py | yolo模型的训练实现 |
| yolov3_to_recognition/utils/callbacks.py | yolo模型的回调函数 |
| yolov3_to_recognition/utils/dataloader.py | yolo模型数据加载器的实现 |
| yolov3_to_recognition/utils/utils_bbox.py | 对yolo模型中的先验框进行解码和预测结果后处理 |
| yolov3_to_recognition/utils/utils_fit.py | yolo模型的训练的工具函数 |


# 对各个文件的简要概述
## face_recognition_and_verification.py

该文件实现了一个简单的人脸识别及人脸验证的流程，先使用yolov3算法对一组照片进行人脸识别，然后保存检测结果，之后使用siamese算法计算图片与参考照片的相似度，并最终得到相似度最高的人脸验证结果。最后在图像上绘制出人脸识别和人脸验证的结果，包括人脸边框和相似度。

## siamese_to_verification\predict.py

该文件是一个人脸验证模型的预测程序。其使用Siamese模型对指定照片与文件夹下的多张照片进行比较，输出各照片与指定照片的相似度。代码中还包括一些异常处理和文件名排序代码。该文件可直接运行，输出结果类似于如下格式：

1, Similarity:0.987

2, Similarity:0.748

3, Similarity:0.621

......

## siamese_to_verification\siamese.py

该文件为Siamese神经网络模型相关文件，用于实现图像相似度比较。其中包含以下主要功能：
- 初始化模型
- 载入预训练权值
- 对输入图像进行不失真的resize
- 归一化处理
- 对预处理后的图像进行相似度比较并输出结果

## siamese_to_verification\train.py

该文件是一个Siamese神经网络的训练脚本。该网络用于人脸验证，并且可以输入图像大小，数据集存放路径，预训练模型路径和批量大小等参数。它将加载Siamese神经网络，设置优化器和学习率调整器，按照训练集和验证集的比例将数据集分为训练集和验证集，以用于训练和测试。脚本将周期性地调整学习率并训练网络。

## siamese_to_verification\utils\dataloader.py

该程序文件为提供了一个用于Siamese神经网络训练和验证的数据集加载和预处理模块。该模块将数据集中的所有图像进行打乱，并将其分为训练集和验证集。使用该模块可以生成随机的类相似和类不相似对，并进行数据增强（图像大小调整、图像翻转、图像放置、图像旋转、色域扭曲等）。最后将处理后的图像转化为张量进行神经网络的训练和验证，同时该文件还提供了一个用于在DataLoader中使用的数据处理函数dataset_collate。

## siamese_to_verification\utils\utils_fit.py

该程序主要实现了训练和验证一个深度学习模型的逻辑，包括计算损失函数、更新网络参数、计算准确率以及保存训练结果等功能。其中，该程序引用了PyTorch内置的神经网络库、PyTorch手动获取的学习率函数以及第三方工具`tqdm`和Pillow。主要实现了`fit_one_epoch()`函数，该函数接受模型、训练集和验证集数据、优化器、损失函数等参数，并完成一轮训练和一轮验证的计算与更新。

## yolov3_to_recognition\Predict.py

这是一个用于人脸识别和图片目标识别的Python程序文件。其中定义了一个名为face_to_recognition的函数，用于处理人脸识别功能，同时提供了一个用于单独运行的if语句块，以便进行图片目标识别。该程序通过调用yolo.py文件中定义的YOLO类实现目标检测，也涉及到一些文件和文件夹的操作。

## yolov3_to_recognition\train.py

该文件是一个用于训练YOLOv3模型的Python程序，将检测到的物体进行分类识别。它利用PyTorch框架和预处理方法，读取训练集和验证集的注释文件，并利用包含分类标签和锚定框的相关文件。该程序将数据加载到模型中并使用训练数据来更新网络参数，同时在每个新epoch之后计算并记录损失。程序分为冻结阶段和解冻阶段，训练分别在两个阶段进行，并包含不同的参数。

## yolov3_to_recognition\voc_annotation.py

这是一段用于生成训练数据集和验证数据集的代码（train.txt 和 val.txt），代码中通过解析 VOC 数据集注释数据里的 XML 文件来获取图像的标注信息（class 和 bounding box），并将信息写入到 txt 文件中。该代码适用于 YOLOv3 神经网络的训练。主要包括读取类别列表，迭代处理数据集，将标注信息写入 txt 文件等功能。注意，在更改文件路径后需要重新运行该文件以更新 train.txt 和 val.txt，而且其中要求 txt 文件的路径不能包含特殊字符。

## yolov3_to_recognition\wideface_to_voc.py

这是一个用于将一个名为wideface的数据集转换为目标检测算法需要的数据集格式的Python脚本。该脚本利用skimage进行图像读取，并根据读入图片的尺寸生成XML文件存储目标信息。目标信息通过WideFace数据集下的wider_face_train_bbx_gt.txt和wider_face_val_bbx_gt.txt获取，转换后的数据集存放在固定的voc数据集路径下，方便后续使用目标检测算法进行训练。

## yolov3_to_recognition\yolo.py

这个文件实现了一个基于yolov3的目标检测函数detect_image，能够在输入一张图像后输出检测到的目标位置，并用方框框出来。实现细节包括但不限于：加载模型权值、预测目标位置、过滤低置信度框、应用非极大抑制、绘制方框和标注文字等。文件包含YOLO类和几个辅助函数。其中YOLO类是主要实现，提供接口函数detect_image来对图像进行目标检测。类初始化时需要指定模型权值文件路径、已知的类别和先验框大小。类主要实现模型加载、数据预处理、模型输出解码、非极大抑制和框绘制等功能。

## yolov3_to_recognition\nets\yolo_training.py

这是一个 YOLOv3 对象检测算法的损失函数实现，该函数计算模型的分类损失和回归损失，并根据真实框和先验框的重合度来调整损失大小。此外，为了增强小目标损失权重和降低大目标损失权重，该损失函数也对大小目标给予不同的处理。

## yolov3_to_recognition\utils\callbacks.py

这是一个Python源代码文件，文件名为callbacks.py，位于yolov3_to_recognition\utils目录下。该文件实现了一个名为LossHistory的类，用于记录训练过程中的损失(loss)和验证损失(val_loss)。另外还包括了一个loss_plot方法，用于绘制损失曲线图，并将图表保存为PNG格式。

## yolov3_to_recognition\utils\dataloader.py

该文件是用于加载数据集的Python程序。其中包括一个继承了torch.utils.data.dataset.Dataset的YoloDataset类和一个用于collate的yolo_dataset_collate函数。YoloDataset类的构造函数需要传入annotation_lines、input_shape、num_classes、train等参数，并定义了__len__和__getitem__方法，其中__getitem__方法用于读取数据集中的图片和标注框，并进行数据增强。yolo_dataset_collate函数用于将数据集中的数据进行打包处理。该程序使用了OpenCV、numpy和PIL库。

## yolov3_to_recognition\utils\utils_bbox.py

该程序文件是一个工具文件，主要实现了对先验框的解码、预测框筛选、去除灰条等功能。其中主要的类是DecodeBox，包含了对先验框进行解码的功能。另外，还包含了对预测结果进行非极大值抑制的功能。

## yolov3_to_recognition\utils\utils_fit.py

此程序文件为yolo模型的训练工具文件，包含了模型训练过程中的训练和验证过程中的前向传播，损失计算和反向传播等步骤，并且定义了一些辅助函数用于获取类别名称和锚点等信息。该文件具有较强的复用性，并且可以通过修改一些超参数来适应不同的数据集和模型。


## 参考资料

https://github.com/bubbliiiing/yolo3-keras

https://github.com/bubbliiiing/Siamese-pytorch

