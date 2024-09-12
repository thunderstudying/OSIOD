import itertools
import random
import os
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager

from detectron2.utils.store_non_list import Store

VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]

VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


T1_CLASS_NAMES = [
    "bird", "cat", "cow", "dog", "horse", "sheep", "elephant", "bear", "zebra", "giraffe",
    "person",
    "bottle", "fork", "cup", "knife", "spoon", "bowl", "wine glass"
]
T2_CLASS_NAMES = [
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "truck", "airplane", "bicycle", "car", "boat", "bus", "motorcycle", "train",
    "cell phone", "laptop", "mouse", "remote", "tv", "keyboard"
]
T3_CLASS_NAMES = [
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
T4_CLASS_NAMES = [
    "chair", "couch", "dining table", "potted plant", "toilet", "bed",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard" "tennis racket"
]

UNK_CLASS = ["unknown"]

# Change this accodingly for task t[2/3/4]
known_classes = list(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES))
# known_classes = list(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES))
# known_classes = list(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES))
root = 'Main/'
train_files = [root+'t1_train.txt', root+'t2_train.txt']
# train_files = [root+'t1_train.txt', root+'t2_train.txt', root+'t3_train.txt']
# train_files = [root+'t1_train.txt', root+'t2_train.txt', root+'t3_train.txt', root+'t4_train.txt']

annotation_location = "COCO2017/Annotations/"

items_per_class = 100
dest_file = root + 't2_ft_' + str(items_per_class) + '.txt'
# dest_file = root + 't3_ft_' + str(items_per_class) + '.txt'
# dest_file = root + 't4_ft_' + str(items_per_class) + '.txt'

file_names = []
for tf in train_files:
    with open(tf, mode="r") as myFile:
        file_names.extend(myFile.readlines())

random.shuffle(file_names)

image_store = Store(len(known_classes), items_per_class)

current_min_item_count = 0

for fileid in file_names:
    fileid = fileid.strip()
    anno_file = os.path.join(annotation_location, fileid + ".xml")

    with PathManager.open(anno_file) as f:
        tree = ET.parse(f)

    for obj in tree.findall("object"):
        cls = obj.find("name").text
        if cls in known_classes:
            image_store.add((fileid,), (known_classes.index(cls),))

    current_min_item_count = min([len(items) for items in image_store.retrieve(-1)])
    print(current_min_item_count)
    if current_min_item_count == items_per_class:
        break

filtered_file_names = []
for items in image_store.retrieve(-1):
    filtered_file_names.extend(items)

print(image_store)
print(len(filtered_file_names))
print('image num:', len(set(filtered_file_names)))

filtered_file_names = set(filtered_file_names)
filtered_file_names = map(lambda x: x + '\n', filtered_file_names)

with open(dest_file, mode="w") as myFile:
    myFile.writelines(filtered_file_names)

print('Saved to file: ' + dest_file)
