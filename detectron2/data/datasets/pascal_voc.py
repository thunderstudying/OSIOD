# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
from fvcore.common.file_io import PathManager
import itertools
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

__all__ = ["load_voc_instances", "register_pascal_voc"]


VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]


VOC_CLASS_NAMES = ["bird", "cat", "cow", "dog", "horse", "sheep", "elephant", "bear", "zebra","giraffe","person" , "bottle", "fork", "cup", "knife", "spoon", "bowl", "wine glass"]

T2_CLASS_NAMES = ["microwave", "oven", "toaster", "sink", "refrigerator","truck", "aeroplane", "bicycle", "car", "boat", "bus", "motorbike", "train" ,"cell phone", "laptop", "mouse", "remote", "tvmonitor", "keyboard"]

T3_CLASS_NAMES = ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench","backpack", "umbrella", "handbag", "tie", "suitcase","book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

T4_CLASS_NAMES = ["chair", "sofa", "diningtable", "pottedplant", "toilet", "bed","banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake","frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"]

UNK_CLASS = ["unknown"]

VOC_COCO_CLASS_NAMES = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))

def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        try:
            with PathManager.open(anno_file) as f:
                tree = ET.parse(f)
        except:
            logger = logging.getLogger(__name__)
            logger.info('Not able to load: ' + anno_file + '. Continuing without aboarting...')
            continue

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls in VOC_CLASS_NAMES_COCOFIED:
                cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_pascal_voc(name, dirname, split, year):
    # if "voc_coco" in name:
    #     class_names = VOC_COCO_CLASS_NAMES
    # else:
    #     class_names = tuple(VOC_CLASS_NAMES)
    class_names = VOC_COCO_CLASS_NAMES
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )
