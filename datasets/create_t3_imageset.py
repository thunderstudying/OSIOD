from pycocotools.coco import COCO
import numpy as np
T3_CLASS_NAMES = [
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Train
coco_annotation_file = 'path/to/instances_train2017.json'
root = "Main/"
dest_file = root + "t3_train_all.txt"

coco_instance = COCO(coco_annotation_file)

image_ids = []
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    if set(classes).issubset(T3_CLASS_NAMES):  # only currently known classes
        image_ids.append(image_details['file_name'].split('.')[0])
        cls.extend(classes)

(unique, counts) = np.unique(cls, return_counts=True)
print({x:y for x,y in zip(unique, counts)})

with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')

print('Created train file')

# Test
coco_annotation_file = 'path/to/instances_val2017.json'
dest_file = root + "t3_test.txt"

coco_instance = COCO(coco_annotation_file)

image_ids = []
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    image_ids.append(image_details['file_name'].split('.')[0])
    cls.extend(classes)

(unique, counts) = np.unique(cls, return_counts=True)
print({x:y for x,y in zip(unique, counts)})

with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')
print('Created test file')
