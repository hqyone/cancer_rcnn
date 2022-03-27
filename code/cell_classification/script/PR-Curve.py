################ 导入相关包 #####################
import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.metrics import scores

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib
from samples.hpv import hpv
import pickle

##############  配置参数  ####
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DATASET_DIR = os.path.join(ROOT_DIR, "datasets/hpv")   #  数据集
config = hpv.NucleusInferenceConfig()
DEVICE = "/cpu:0"
TEST_MODE = "inference"
def get_ax(rows=1, cols=1, size=16):
    fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    fig.tight_layout()
    return ax
def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存txt文件成功")

#####  加载测试集数据  #####
dataset = hpv.NucleusDataset()
dataset.load_nucleus(DATASET_DIR, "stage1_test")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

#####  导入模型  ####
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",model_dir=LOGS_DIR,config=config)
weights_path = "/Mask_RCNN/logs/hpv20210105T1517/mask_rcnn_hpv_0799.h5"
# weights_path = "/Mask_RCNN/logs/nucleus20210105T1517/mask_rcnn_nucleus_0285.h5"
model.load_weights(weights_path, by_name=True)
# image_id = random.choice(dataset.image_ids)   # 随机选取一张测试集
image_ids = dataset.image_ids

APs = []
count1 = 0
for image_id in image_ids:
    info = dataset.image_info[image_id]
    print("image_id: ", image_id)
    # ####重要步骤：获得测试图片的信息
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    # ###保存实际结果
    if count1 == 0:
        gt_bbox_all, gt_class_id_all, gt_mask_all = gt_bbox, gt_class_id, gt_mask
    else:
        gt_bbox_all = np.concatenate((gt_bbox_all, gt_bbox), axis=0)
        gt_class_id_all = np.concatenate((gt_class_id_all, gt_class_id), axis=0)
        gt_mask_all = np.concatenate((gt_mask_all, gt_mask), axis=2)
    # # 显示检测结果
    results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)
    r = results[0]
    # 保存预测结果
    if count1 == 0:
        pre_masks_all, pre_class_ids_all, pre_rois_all, pre_scores_all = r['masks'], r["class_ids"], r["rois"], r["scores"]
    else:
        pre_masks_all = np.concatenate((pre_masks_all, r['masks']), axis=2)
        pre_class_ids_all = np.concatenate((pre_class_ids_all, r['class_ids']), axis=0)
        pre_rois_all = np.concatenate((pre_rois_all, r['rois']), axis=0)
        pre_scores_all = np.concatenate((pre_scores_all, r['scores']), axis=0)

    count1 += 1


# # 在阈值0.5到0.95之间每隔0.1显示AP值
# utils.compute_ap_range(gt_bbox_all, gt_class_id_all, gt_mask_all, pre_rois_all, pre_class_ids_all, pre_scores_all, pre_masks_all, verbose=1)
## 在图片中显示真实与预测之间的差异
# visualize.display_differences(image, gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'],
#                               dataset.class_names, ax=get_ax(), show_box=False, show_mask=False, iou_threshold=0.5, score_threshold=0.5)
# plt.show()

# ######绘制PR曲线######
AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox_all, gt_class_id_all, gt_mask_all,
                                                     pre_rois_all, pre_class_ids_all, pre_scores_all, pre_masks_all)
print("precisions: ", precisions)
print("AP: ", AP)

plt.figure("P-R Curve")
plt.title('Precision-Recall Curve. AP@50 = {:.3f}'.format(AP))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recalls, precisions, 'b', label='PR')
plt.show()
text_save('Kpreci.txt', precisions)
text_save('Krecall.txt', recalls)