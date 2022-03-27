import os
import numpy as np
from skimage.draw import polygon
import cv2
import json
from shutil import copyfile
from pathlib import Path
from skimage.measure import find_contours
import glob
import base64
import skimage
from mrcnn import utils
from mrcnn import visualize
import matplotlib.pyplot as plt
import glob
import random 

import tensorflow as tf
import keras

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


def getFileNameInfor(filepath):
    file_name = filepath.split('/')[-1]
    file_base = Path(filepath).stem
    fdir = os.path.dirname(os.path.abspath(filepath))
    return (fdir, file_name, file_base)

def boundary2Mask(xs, ys, img):
    mask = np.zeros(img.shape[0:2])
    rr, cc = polygon(xs, ys, img.shape)
    mask[rr, cc] = 255
    return mask

def mask2Boundary(mask, sample_ratio=5):
    contours = find_contours(mask, 200)
    boundaries=[]
    for i, verts in enumerate(contours):
        new_vects = []
        for k, p in enumerate(verts):
            # Subtract the padding and flip (y, x) to (x, y)z
            if k % sample_ratio==0:
                new_vects.append(list(p))
        boundaries.append(new_vects)
    return boundaries

def maskFile2Boundary(mask_file, sample_ratio=5):
    mask = cv2.imread(mask_file)[:,:,0] 
    mask2Boundary(mask, sample_ratio=sample_ratio)

def get_iou(roi1, roi2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1={"y1":roi1[0], 'x1':roi1[1],"y2":roi1[2], 'x2':roi1[3]}
    bb2={"y1":roi2[0], 'x1':roi2[1],"y2":roi2[2], 'x2':roi2[3]}
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# Remove overlapping object from predict output
def removeDuplicateObject(r, min_ratio=0.7):
    classid_ls = list(map(int, r['class_ids']))
    scores = r["scores"]
    masks = r["masks"]
    rois =r['rois']

    new_rois_ls = []
    new_classid_ls = []
    new_scores = []
    new_masks = None
    removed_indexes = []
    for i in range(len(classid_ls)-1):
        for k in range(i+1, len(classid_ls)):
            if k in removed_indexes:
                continue
            mask_a = np.reshape(masks[:,:,i],(masks.shape[0],masks.shape[1],1))
            roi_a = rois[i]
            class_a = classid_ls[i]
            mask_b = np.reshape(masks[:,:,k],(masks.shape[0],masks.shape[1],1))
            roi_b = rois[k]
            class_b = classid_ls[k]
            #print(utils.compute_overlaps_masks(mask_a, mask_b))
            #print('as')
            if get_iou(roi_a,roi_b)>0.8:
            #if utils.compute_overlaps_masks(mask_a, mask_b)[0]>min_ratio:
                if class_a<class_b:
                    removed_indexes.append(k)
                else:
                    removed_indexes.append(i)
                continue
        if i not in removed_indexes:
            new_rois_ls.append(rois[i])
            new_classid_ls.append(classid_ls[i])
            new_scores.append(scores[i])
            if new_masks is None:
                new_masks=np.reshape(masks[:,:, i],(masks.shape[0],masks.shape[1],1))
            else:
                mask=np.reshape(masks[:,:, i],(masks.shape[0],masks.shape[1],1))
                new_masks=np.concatenate([new_masks, mask], axis=-1)
    return {'rois':new_rois_ls, 'class_ids':new_classid_ls, "scores":new_scores, "masks":new_masks}

# Search all images in image_dir and use the model to do predict to get json file with image whichi is ready for labelme to read and edit
def predictDirToJson(image_dir, model, out_dir, ext="*.png", class_names=['BG', 'yin', 'yin-yang', 'yang']):
    for f in glob.glob(f'{image_dir}/**/{ext}', recursive=True):
        if "masks" not in f:
            predictImageToJson(f,model,out_dir,class_names = class_names)

# store the predict results into json file which can be readed by Labelme
# className = ['BG', 'yin', 'yin-yang', 'yang']   # 把数字对应上标签
def predictImageToJson(image_path, model,  out_dir, class_names=['BG', 'yin', 'yin-yang', 'yang']):
    img_dir, img_name, img_id = getFileNameInfor(image_path)
    image = skimage.io.imread(image_path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    imageHeight = image.shape[0]    # 获得长宽
    imageWidth = image.shape[1]
    rr = model.detect([image], verbose=1)[0]
    r = removeDuplicateObject(rr,0.6)
    with open(image_path, 'rb') as f:   # 编码图片 base64
        imageData = base64.b64encode(f.read())
        imageData = str(imageData, encoding='utf-8')
    json_data = {
        'version': "4.5.6",
        "flags": {},
        'shapes': [],
        'imagePath': img_dir,
        'imageData': imageData,
        'imageHeight': imageHeight,
        'imageWidth': imageWidth
    }
    for i in range(0, len(r['class_ids'])):
        class_id = int(r['class_ids'][i])
        score = int(r["scores"][i])
        mask = r["masks"][:, :, i]*255
        boundarys = mask2Boundary(mask)
        obj = {
            'ID': i,
            'label': class_names[class_id],
            'points': [],
            'group_id': None,
            'shape_type': "polygon",
            'flags': {},
        }
        i = 0
        for boundary in boundarys:
            for point in boundary:
                obj["points"].append([float(point[1]), float(point[0])])
        json_data['shapes'].append(obj)
    json_str = json.dumps(json_data, indent=4)

    out_dir=f"{out_dir}/{img_id}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open('{}/{}.json'.format(out_dir, img_id), 'w') as json_file:
        json_file.write(json_str)
    copyfile(image_path, '{}/{}.png'.format(out_dir, img_id))
    # Visualized the labeling results
    visualize.display_instances(
            image, rr['rois'], rr['masks'], rr['class_ids'],
            class_names, rr['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
    plt.savefig("{}/{}_label.jpg".format(out_dir, img_id))


# The function is 
def json2masks(img_file):
    img_dir, img_name, image_id = getFileNameInfor(img_file)
    json_file = f"{img_dir}/{image_id}.json"
    if os.path.isfile(json_file):
        img = cv2.imread(img_file)
        with open(json_file) as JSON:
            data = json.load(JSON)
            image_name = data['imagePath']
            image_height = data['imageHeight']
            image_width = data['imageWidth']
            shapes = data["shapes"]
            if len(shapes)>0:
                # mod_image_id=image_id.replace(" ","").replace("(","_").replace(")","")
                os.makedirs(f'{img_dir}/masks', exist_ok=True)
                for f in os.listdir(f'{img_dir}/masks'):
                    os.remove(os.path.join(f'{img_dir}/masks', f))
                for i, s in enumerate(shapes):
                    label = s['label']
                    points = s['points']
                    xs = []
                    ys = []
                    for p in points:
                        ys.append(p[0])
                        xs.append(p[1])
                    if len(xs)>0 and len(ys)>0:
                        mask = boundary2Mask(xs, ys, img)
                        cv2.imwrite(f"{img_dir}/masks/{i}_{label}.png", mask)

# Search all images in image_dir and use the model to do predict to get json file with image whichi is ready for labelme to read and edit
def json2masksInDirectory(image_dir, ext="*.png"):
    for f in glob.glob(f'{image_dir}/**/{ext}', recursive=True):
        if "masks" not in f:
            json2masks(f)

def getTrainValImageIDs(dataset_dir,val_number,ext="*.png"):
    """Load a subset of the nuclei dataset.

    dataset_dir: Root directory of the dataset
    subset: Subset to load. Either the name of the sub-directory,
            such as stage1_train, stage1_test, ...etc. or, one of:
            * train: stage1_train excluding validation images
            * val: validation images from VAL_IMAGE_IDS
    """
    # Add classes. We have one class.
    # Naming the dataset nucleus, and the class nucleus
    

    # Which subset?
    # "val": use hard-coded list above
    # "train": use data from stage1_train minus the hard-coded list above
    # else: use the data from the specified sub-directory
    # assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
    #subset_dir = "stage1_train" if subset in ["train", "val"] else subset
    val_Image_IDS=[]
    training_Image_IDS=[]
    Image_IDS=[]
    for f in glob.glob(f'{dataset_dir}/**/{ext}', recursive=True):
        if "masks" not in f:
            img_dir, img_name, img_id = getFileNameInfor(f)
            Image_IDS.append(img_id)
    if val_number > len(Image_IDS)/2:
        raise ValueError(f'The validate data number {val_number} is two big to samples number {len(Image_IDS)}')
    val_Image_IDS = random.choices(Image_IDS, k=val_number)
    training_Image_IDS = list(set(Image_IDS) - set(val_Image_IDS))
    return( training_Image_IDS, val_Image_IDS)