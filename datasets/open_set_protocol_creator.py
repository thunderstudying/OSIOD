import json
from pycocotools.coco import COCO

COCOID_to_pascalID = {
    5: 1,
    2: 2,
    16: 3,
    9: 4,
    44: 5,
    6: 6,
    3: 7,
    17: 8,
    62: 9,
    21: 10,
    67: 11,
    18: 12,
    19: 13,
    4: 14,
    1: 15,
    64: 16,
    20: 17,
    63: 18,
    7: 19,
    72: 20
}
coco_ids, pascal_ids = zip(*COCOID_to_pascalID.items())
PascalID_to_COCOID = dict(zip(pascal_ids, coco_ids))

coco_2017_training_file = 'path/to/instances_train2017.json'
pascal_2007_test_file = 'path/to/pascal_test2007.json'

coco_2017_train = COCO(coco_2017_training_file)
pascal_2007_test = COCO(pascal_2007_test_file)

categories_info = []
COCO_new_class_id_mapping = {}
PASCAL_new_class_id_mapping = {}
for new_class_id, (COCO_ID, PASCAL_ID) in enumerate(COCOID_to_pascalID.items(), start=1):
    COCO_new_class_id_mapping[COCO_ID] = new_class_id
    PASCAL_new_class_id_mapping[PASCAL_ID] = new_class_id
    categories_info.append(pascal_2007_test.cats[PASCAL_ID])

test_categories_info = categories_info[:]
test_categories_info.append({'supercategory': 'none', 'id': -1, 'name': 'unknown'})


# Making the protocol for only knowns
def pascal_to_custom_training_protcol(input_file_name, output_file_name, testing_cats=False):
    coco_obj = COCO('../protocol/' + input_file_name)
    new_json = {}
    new_json['images'] = []
    new_json['annotations'] = []
    if testing_cats:
        new_json['categories'] = test_categories_info[:]
    else:
        new_json['categories'] = categories_info[:]

    for img_id in coco_obj.imgs.keys():
        new_json['images'].append(coco_obj.imgs[img_id])
        ann_ids = coco_obj.getAnnIds(imgIds=[img_id])
        for annotation in coco_obj.loadAnns(ids=ann_ids):
            annotation['category_id'] = PASCAL_new_class_id_mapping[annotation['category_id']]
            difficult = annotation['ignore']
            del annotation['ignore']
            if difficult == 1:
                continue
            new_json['annotations'].append(annotation)
    json.dump(new_json,
              open('../protocol/custom_protocols/' + output_file_name,
                   "w"))

pascal_orignal_files = ('voc_2007_train.json', 'voc_2007_val.json', 'voc_2012_train.json', 'voc_2012_val.json')
updated_files = ['no_difficult_custom_' + _ for _ in pascal_orignal_files]
for input_file_name, output_file_name in zip(pascal_orignal_files, updated_files):
    pascal_to_custom_training_protcol(input_file_name, output_file_name)
pascal_to_custom_training_protcol('voc_2007_test.json', 'custom_voc_2007_test_no_difficult.json', testing_cats=True)

# Making the protocol for mixed unknowns
images_with_knowns = []
for _ in COCOID_to_pascalID.keys():
    images_with_knowns.extend(coco_2017_train.getImgIds(catIds=[_]))
images_with_knowns = set(images_with_knowns)
images_without_knowns = set(coco_2017_train.imgs) - images_with_knowns

json_data = coco_2017_train.dataset.copy()
new_json = {}
new_json['categories'] = test_categories_info[:]
new_json['images'] = []
new_json['annotations'] = []

# add an equal number of images with only unknown-class instances from the COCO 2017 training set into VOC images
WR1_mixed_unknowns_file_name = 'WR1_Mixed_Unknowns.json'

for img_id in images_without_knowns:
    new_json['images'].append(coco_2017_train.imgs[img_id])
    ann_ids = coco_2017_train.getAnnIds(imgIds=[img_id])
    for annotation in coco_2017_train.loadAnns(ids=ann_ids):
        annotation['category_id'] = -1
        new_json['annotations'].append(annotation)

json.dump(new_json, open(f"../protocol/custom_protocols/{WR1_mixed_unknowns_file_name}", "w"))
