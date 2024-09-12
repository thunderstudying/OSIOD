from pycocotools.coco import COCO
import json

CLASS_NAMES = [
    'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'elephant', 'bear', 'zebra', 'giraffe',
    'person',
    'bottle', 'fork', 'cup', 'knife', 'spoon', 'bowl', 'wine glass',

    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'truck', 'airplane', 'bicycle', 'car', 'boat', 'bus', 'motorcycle', 'train',
    'cell phone', 'laptop', 'mouse', 'remote', 'tv', 'keyboard',

    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',

    'chair', 'couch', 'dining table', 'potted plant', 'toilet', 'bed',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'
]

filter_txt = r'Main/t1_train.txt'
# filter_txt = r'Main/t2_train.txt'
# filter_txt = r'Main/t2_ft_100.txt'
# filter_txt = r'Main/t3_train.txt'
# filter_txt = r'Main/t3_ft_100.txt'
# filter_txt = r'Main/t4_train.txt'
# filter_txt = r'Main/t4_ft_100.txt'

coco_annotation_file = 'path/to/instances_train2017.json'

root = "../protocol/custom_protocols/"  # save root dir

c_id_map = {}
coco_instance = COCO(coco_annotation_file)
with open(coco_annotation_file, 'r')as f:
    data = json.load(f)

with open(filter_txt, 'r') as f2:
    img_names = f2.readlines()
img_names = [x.strip() for x in img_names]

images_list = []
for image in data['images']:
    file_name = image['file_name'].split('.')[0]
    if file_name in img_names:
        images_list.append(image)

print('train image:', len(images_list))
data['images'] = images_list

print('categories:', data['categories'])

for cate in data['categories']:
    cls = cate['name']
    old_id = cate['id']
    new_id = CLASS_NAMES.index(cls)+1
    cate['supercategory'] = 'none'
    if new_id <= 18:  # t1
    # if 19 <= new_id <= 37:  # t2
    # if new_id <= 37:  # t2_ft
    # if 38 <= new_id <= 54:  # t3
    # if new_id <= 54:  # t3_ft
    # if 55 <= new_id <= 80:  # t4
    # if new_id <= 80:  # t4_ft
        cate['id'] = new_id
        c_id_map[old_id] = new_id
    else:
        cate['id'] = -1
        c_id_map[old_id] = -1
        cate['name'] = 'unknown'

data['categories'] = sorted(data['categories'], key=lambda x:x['id'])[62:]    # t1   18 classes
# data['categories'] = sorted(data['categories'], key=lambda x:x['id'])[61:]  # t2   19 classes
# data['categories'] = sorted(data['categories'], key=lambda x:x['id'])[43:]  # t2ft 37 classes
# data['categories'] = sorted(data['categories'], key=lambda x:x['id'])[63:]  # t3   17 classes
# data['categories'] = sorted(data['categories'], key=lambda x:x['id'])[26:]  # t3ft 54 classes
# data['categories'] = sorted(data['categories'], key=lambda x:x['id'])[54:]  # t4   26 classes
# data['categories'] = sorted(data['categories'], key=lambda x:x['id'])       # t4ft 80 classes

print('new categories', data['categories'])
print(c_id_map)

res = json.dumps(data)

with open(root+'OSIOD_t1.json', 'w') as f:
# with open(root+'OSIOD_t2.json','w') as f:
# with open(root+'OSIOD_t2_ft.json','w') as f:
# with open(root+'OSIOD_t3.json','w') as f:
# with open(root+'OSIOD_t3_ft.json','w') as f:
# with open(root+'OSIOD_t4.json','w') as f:
# with open(root+'OSIOD_t4_ft.json','w') as f:
    f.write(res)
