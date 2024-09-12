import sys
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter
import random

root = "Main/"

t1_train = root + "t1_train_all.txt"
t1_val_file = root + "t1_val.txt"
t2_train = root + "t2_train_all.txt"
t2_val_file = root + "t2_val.txt"
t3_train = root + "t3_train_all.txt"
t3_val_file = root + "t3_val.txt"
t4_train = root + "t4_train_all.txt"
t4_val_file = root + "t4_val.txt"
all_task_val = root + "all_task_val.txt"
t1_test = root + "t1_test.txt"
t2_test = root + "t2_test.txt"
t3_test = root + "t3_test.txt"
t4_test = root + "t4_test.txt"
all_task_test = root + "all_task_test.txt"
t1_new_file = root + "t1_train.txt"
t2_new_file = root + "t2_train.txt"
t3_new_file = root + "t3_train.txt"
t4_new_file = root + "t4_train.txt"

# Val set creation
with open(t1_train, 'r') as f:
    t1_list = [line.strip() for line in f.readlines()]
t1_val = random.sample(t1_list, 1000)  # t1
print("t1_val:", len(t1_val))
with open(t1_val_file, 'w') as file:
    for image_id in t1_val:
        file.write(str(image_id) + '\n')
print('Created t1_val file')

with open(t2_train, 'r') as f:
    t2_list = [line.strip() for line in f.readlines()]
t2_val = random.sample(t2_list, 1000)  # t2
print("t2_val:", len(t2_val))
with open(t2_val_file, 'w') as file:
    for image_id in t2_val:
        file.write(str(image_id) + '\n')
print('Created t2_val file')

with open(t3_train, 'r') as f:
    t3_list = [line.strip() for line in f.readlines()]
t3_val = random.sample(t3_list, 1000)  # t3
print("t3_val:", len(t3_val))
with open(t3_val_file, 'w') as file:
    for image_id in t1_val:
        file.write(str(image_id) + '\n')
print('Created t3_val file')

with open(t4_train, 'r') as f:
    t4_list = [line.strip() for line in f.readlines()]
t4_val = random.sample(t4_list, 1000)  # t4
print("t4_val:", len(t4_val))
with open(t4_val_file, 'w') as file:
    for image_id in t4_val:
        file.write(str(image_id) + '\n')
print('Created t4_val file')

# all_task_val
val_list = t1_val + t2_val + t3_val + t4_val  # t1+t2+t3+t4
with open(all_task_val, 'w') as file:
    for image_id in val_list:
        file.write(str(image_id) + '\n')
print('Created all_task_val file')

# all_task_test
with open(t1_test, 'r') as f:
    t1_test_list = f.read().splitlines()
with open(t2_test, 'r') as f:
    t2_test_list = f.read().splitlines()
with open(t3_test, 'r') as f:
    t3_test_list = f.read().splitlines()
with open(t4_test, 'r') as f:
    t4_test_list = f.read().splitlines()
test_list = t1_test_list + t2_test_list + t3_test_list + t4_test_list
with open(all_task_test, 'w') as f:
    for image_id in list(set(test_list)):
        f.write(str(image_id) + '\n')
print('Created all_task_test file')

# Training set deduplication
with open(t1_train, 'r') as t1_file:
    t1_list = t1_file.read().splitlines()
print("t1_list:", len(t1_list))
t1_train = [x for x in t1_list if x not in val_list]
print("t1_train:", len(t1_train))
with open(t1_new_file, 'w') as file:
    for image_id in t1_train:
        file.write(str(image_id) + '\n')
print('Created file')

with open(t2_train, 'r') as t2_file:
    t2_list = t2_file.read().splitlines()
print("t2_list:", len(t2_list))
t2_train = [x for x in t2_list if x not in val_list]
print("t2_train:", len(t2_train))
with open(t2_new_file, 'w') as file:
    for image_id in t2_train:
        file.write(str(image_id) + '\n')
print('Created file')

with open(t3_train, 'r') as t3_file:
    t3_list = t3_file.read().splitlines()
print("t3_list:", len(t3_list))
t3_train = [x for x in t3_list if x not in val_list]
print("t3_train:", len(t3_train))
with open(t3_new_file, 'w') as file:
    for image_id in t3_train:
        file.write(str(image_id) + '\n')
print('Created file')

with open(t4_train, 'r') as t4_file:
    t4_list = t4_file.read().splitlines()
print("t4_list:", len(t4_list))
t4_train = [x for x in t4_list if x not in val_list]
print("t4_train:", len(t4_train))
with open(t4_new_file, 'w') as file:
    for image_id in t4_train:
        file.write(str(image_id) + '\n')
print('Created file')

with open(all_task_val, 'r') as val_file:
    val_list = val_file.read().splitlines()
print("val_list:", len(val_list))

with open(all_task_test, 'r') as test_file:
    test_list = test_file.read().splitlines()
print("test_list:", len(test_list))
