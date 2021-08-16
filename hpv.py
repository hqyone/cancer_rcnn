# """
# Mask R-CNN
# Train on the nuclei segmentation dataset from the
# Kaggle 2018 Data Science Bowl
# https://www.kaggle.com/c/data-science-bowl-2018/
#
# Licensed under the MIT License (see LICENSE for details)
# Written by Waleed Abdulla
#
# ------------------------------------------------------------
#
# Usage: import the module (see Jupyter notebooks for examples), or run from
#        the command line as such:
#
#     # Train a new model starting from ImageNet weights
#     python3 lung.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
#
#     # Train a new model starting from specific weights file
#     python3 lung.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5
#     python nucleus_li.py train --dataset=F:/stage1_train --subset=train --weights=C:/Users/LXG/Mask_RCNN/resnet50.h5
#     # Resume training a model that you had trained earlier
#     python3 lung.py train --dataset=/path/to/dataset --subset=train --weights=last
#
#     # Generate submission file
#     python3 lung.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
# """

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
# train:python hpv.py train --dataset=F:/Mask_RCNN/datasets/hpv --subset=stage1_train --weights=last
#  python hpv.py train --dataset=F:/Mask_RCNN/datasets/hpv --subset=stage1_train --weights=F:/Mask_RCNN/logs/hpv20210105T1517/mask_rcnn_hpv_0600.h5
if __name__ == '__main__':
    import matplotlib

    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

import tensorflow as tf
import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

ROOT_DIR = os.path.abspath("../../")  # 指定根目录

# 导入Mask RCNN
sys.path.append(ROOT_DIR)  # 查找库的本地版本
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import time
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "F:/Mask_RCNN/logs/hpv20210105T1517/mask_rcnn_hpv_0600.h5")

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

RESULTS_DIR = os.path.join(ROOT_DIR, "results/hpv/")
# RESULTS_DIR = os.path.join(ROOT_DIR, "logs")   # 指定训练好的权重路径
# 是否可以  RESULTS_DIR = os.path.join(F:\0_LXG\Mask_RCNN,"logs")
# 或者  RESULTS_DIR = 'F:\0_LXG\Mask_RCNN'


# VAL_IMAGE_IDS = ["0","1","2",]  # 指定验证集照片
# VAL_IMAGE_IDS=[]
# VALPath = r'F:\Mask_RCNN\datasets\hpv\val'
#
# for root,dirs,files in os.walk(VALPath):
#
#     for dir in dirs:
#         if 'images' in dir:
#            val=root.split("\\")[-1]
#            VAL_IMAGE_IDS.append("\""+val+"\"")

VAL_IMAGE_IDS = [
    "0",
    "1",
    "2",
    "3",
    "4"
    "5",
    "6",
    "7",
    "8", "9", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100",
    "200", "210", "220", "230", "240", "250", "260", "270", "280", "290", "300",
    "310", "320", "330", "340", "350", "360", "370", "380", "390", "400",
    "410", "420", "430", "440", "450", "460", "470", "471", "472", "473", "499"
]
# VAL_IMAGE_IDS = []
# for i in range(0, 950):
#     if i % 6 == 0:
#         VAL_IMAGE_IDS.append(str(i))

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "hpv"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + yin + yin-yang + yang

    #     STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    #     VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    STEPS_PER_EPOCH = (500) // IMAGES_PER_GPU  # 依据图片数量分配GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    DETECTION_MIN_CONFIDENCE = 0

    BACKBONE = "resnet50"  # 选择网络结构类型

    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 1024  # 调整图片大小  1024x1024
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 2.0

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # 方块锚的像素大小

    #  非最大抑值ROI
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    RPN_NMS_THRESHOLD = 0.9

    #   每个图像要用于RPN训练的锚点数
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    #  图片通道平均值
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # 图像分辨率较高时 将掩码大小降低
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128  # 每幅图像可输入到mask head 中的 ROI数

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400  # 每幅图像最终检测的最大数量


class NucleusInferenceConfig(NucleusConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # 图像分配的GPU
    IMAGE_RESIZE_MODE = "pad64"  # ???
    RPN_NMS_THRESHOLD = 0.7


#################
# 加载数据
################
class NucleusDataset(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "yin")
        self.add_class("nucleus", 2, "yin-yang")
        self.add_class("nucleus", 3, "yang")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            image_ids = next(os.walk(dataset_dir))[1]

            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))
                # image_ids = list(set(image_ids))

        # Add images
        for image_id in image_ids:
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = []
        labels = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
                labels.append(int(f.split('.')[0].split('_')[1]))
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        # return mask, np.ones([mask.shape[-1]], dtype=np.int32)
        return mask, np.array(labels, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = NucleusDataset()
    dataset_train.load_nucleus(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleusDataset()
    dataset_val.load_nucleus(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=720,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=820,
                augmentation=augmentation,
                layers='all')


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset()
    dataset.load_nucleus(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        print(image_id)
        # picture = "D:\\Desktop\\hpv-test\\test\\xx_best_425_31_36.png"
        # image = skimage.io.imread(picture)
        starttime=time.clock()
        #image = skimage.transform.resize(image, (1024, 1024), preserve_range=True)
        #print("image.shape:", image.shape)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        # rle = mask_to_rle(source_id, r["masks"], r["scores"])
        # print("r[boxes].shape:", r["rois"].shape)
        # print("用时：",time.clock()-starttime)
        # submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))


    # Save to csv file
    # submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    # file_path = os.path.join(submit_dir, "submit.csv")
    # with open(file_path, "w") as f:
    #     f.write(submission)
    # print("Saved to ", submit_dir)

############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NucleusConfig()
    else:
        config = NucleusInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        print("111111111111111111111111111111111111111111111")
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    print(args.weights.lower())
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
        print("222222222222222222222222222222222222222")
    else:
        print("111111111111111111111111111111111111111111111")
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))