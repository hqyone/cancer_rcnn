'''
首先导入相关包，然后加载默认配置
'''
import os
import sys
import numpy as np
import tensorflow as tf
import re
ROOT_DIR = os.path.abspath("/Mask_RCNN")
sys.path.append(ROOT_DIR)
# import mrcnn.model as modellib
import mrcnn.model as modellib
from samples.lung import lung
import skimage.io
import skimage.transform
from skimage.measure import find_contours
import cv2
import json
import base64
from mrcnn import visualize
from mrcnn.visualize import display_images

# MODEL_DIR = os.path.join(ROOT_DIR, "logs")
LOGS_DIR = os.path.join(ROOT_DIR, "/logs/nucleus20210105T1517")
DATASET_DIR = os.path.join(ROOT_DIR, "datasets/hpv")

config = lung.NucleusConfig()

config = lung.NucleusInferenceConfig()  # 少继承了一个参数

DEVICE = "/cpu:0"


class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

TEST_MODE = "inference"

'''
调用获得边界函数
'''
def getBoundary(mask):
    padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)z
        verts = np.fliplr(verts) - 1
    return verts


line_color_dic = {
    '1': [255, 0, 0, 128],
    "2": [0, 255, 0, 128],
    "3": [0, 0, 255, 128]
}
fill_color_dic = {
    '1': [255, 0, 0, 128],
    "2": [0, 255, 0, 128],
    "3": [0, 0, 255, 128]
}
'''
加载模型检测目标
'''
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=LOGS_DIR, config=config)
    weights_path = "F:\\Mask_RCNN\\logs\\hpv20210105T1517\\mask_rcnn_hpv_0799.h5"
    model.load_weights(weights_path, by_name=True)

    sourse_dir = "F:\\cervical_cancer\\1111\\stage1_test\\"

    sourse_dir_list = os.listdir(sourse_dir)

    sourse_dir_list.sort()

    sourse_dir_list.sort(key=lambda x:int(x.split('.')[0]))

    img_nums = len(sourse_dir_list)

    t = 0
    for i in range(img_nums):
        image_name = sourse_dir + str(i) + '\\' + 'images' + '\\' + sourse_dir_list[i] + '.png'
        print(image_name)
        imagePath = image_name.split('\\')[-1]
        image = skimage.io.imread(image_name)
        imageHeight = image.shape[0]    # 获得长宽
        imageWidth = image.shape[1]
        r = model.detect([image], verbose=1)[0]
        with open(image_name, 'rb') as f:   # 编码图片 base64
            imageData = base64.b64encode(f.read())
            imageData = str(imageData, encoding='utf-8')
        json_data = {
            'version': "4.5.6",
            "flags": {},
            'shapes': [],
            'imagePath': imagePath,
            'imageData': imageData,
            'imageHeight': imageHeight,
            'imageWidth': imageWidth
        }
        for i in range(0, len(r['class_ids'])):
            class_id = int(r['class_ids'][i])
            className = ['BG', 'yin', 'yin-yang', 'yang']   # 把数字对应上标签
            line_color = line_color_dic[str(int(class_id))]
            fill_color = fill_color_dic[str(int(class_id))]
            score = int(r["scores"][i])
            mask = r["masks"][:, :, i]
            boundary = getBoundary(mask)
            obj = {
                'ID': i,
                'label': className[class_id],
                'points': [],
                'group_id': None,
                'shape_type': "polygon",
                'flags': {},
            }
            i = 0
            for point in boundary:
                i += 1
                if i%8 == 0:
                    obj["points"].append([float(point[0]), float(point[1])])
            json_data['shapes'].append(obj)
        json_str = json.dumps(json_data, indent=4)
        with open('F:/cervical_cancer/1111/image-json' + '/' + str(t) + '.json', 'w') as json_file:
            json_file.write(json_str)
        t += 1
