set -x
set -o nounset
set -o errexit

arch=s

num_gpu=2
num_workers=4
if [ $arch = x ]
then
  ims_per_batch=30  # x
elif [ $arch = l ]
then
  ims_per_batch=40 # l
elif [ $arch = m ]
then
  ims_per_batch=64  # m
else
  ims_per_batch=100  # s
fi

epoch=350

t1_train_image=13372
((t1_per_epoch=t1_train_image/ims_per_batch+1))

t2_train_image=6524
((t2_per_epoch=t2_train_image/ims_per_batch+1))
t2_ft_image=1477
((t2_ft_per_epoch=t2_ft_image/ims_per_batch+1))

t3_train_image=4637
((t3_per_epoch=t3_train_image/ims_per_batch+1))
t3_ft_image=2201
((t3_ft_per_epoch=t3_ft_image/ims_per_batch+1))

t4_train_image=4391
((t4_per_epoch=t4_train_image/ims_per_batch+1))
t4_ft_image=2826
((t4_ft_per_epoch=t4_ft_image/ims_per_batch+1))

((max_iter=epoch*t1_per_epoch))
python main.py --num-gpus ${num_gpu} --config-file training_configs/YOLO-UCD/t1/t1_train.yaml OUTPUT_DIR YOLO-UCD_${arch}/t1 DATALOADER.NUM_WORKERS ${num_workers} MODEL.YOLO.NC 80 MODEL.YOLO.CUR_INTRODUCED_CLS 18 MODEL.YOLO.LABEL_SMOOTHING 0.1 SOLVER.IMS_PER_BATCH ${ims_per_batch} SOLVER.CHECKPOINT_PERIOD ${t1_per_epoch} TEST.EVAL_PERIOD ${t1_per_epoch} SOLVER.MAX_ITER ${max_iter} MODEL.YOLO.ARCH models/yolov5${arch}.yaml

epoch=25
((max_iter=epoch*t2_per_epoch))
python main.py --num-gpus ${num_gpu} --config-file training_configs/YOLO-UCD/t2/t2_train.yaml OUTPUT_DIR YOLO-UCD_${arch}/t2 DATALOADER.NUM_WORKERS ${num_workers} MODEL.YOLO.NC 80 MODEL.YOLO.CUR_INTRODUCED_CLS 19 MODEL.WEIGHTS YOLO-UCD_${arch}/t1/model_final.pth MODEL.YOLO.LABEL_SMOOTHING 0.1 SOLVER.IMS_PER_BATCH ${ims_per_batch} SOLVER.CHECKPOINT_PERIOD ${t2_per_epoch} TEST.EVAL_PERIOD ${t2_per_epoch} SOLVER.MAX_ITER ${max_iter} MODEL.YOLO.ARCH models/yolov5${arch}.yaml

epoch=100
((max_iter=epoch*t2_ft_per_epoch))
python main.py --num-gpus ${num_gpu} --config-file training_configs/YOLO-UCD/t2/t2_ft.yaml OUTPUT_DIR YOLO-UCD_${arch}/t2_ft DATALOADER.NUM_WORKERS ${num_workers} MODEL.YOLO.NC 80 MODEL.YOLO.CUR_INTRODUCED_CLS 37 MODEL.WEIGHTS YOLO-UCD_${arch}/t2/model_final.pth MODEL.YOLO.LABEL_SMOOTHING 0.1 SOLVER.IMS_PER_BATCH ${ims_per_batch} SOLVER.CHECKPOINT_PERIOD ${t2_ft_per_epoch} TEST.EVAL_PERIOD ${t2_ft_per_epoch} SOLVER.MAX_ITER ${max_iter} MODEL.YOLO.ARCH models/yolov5${arch}.yaml

epoch=25
((max_iter=epoch*t3_per_epoch))
python main.py --num-gpus ${num_gpu} --config-file training_configs/YOLO-UCD/t3/t3_train.yaml OUTPUT_DIR YOLO-UCD_${arch}/t3 DATALOADER.NUM_WORKERS ${num_workers} MODEL.YOLO.NC 80 MODEL.YOLO.CUR_INTRODUCED_CLS 17 MODEL.WEIGHTS YOLO-UCD_${arch}/t2_ft/model_final.pth MODEL.YOLO.LABEL_SMOOTHING 0.1 SOLVER.IMS_PER_BATCH ${ims_per_batch} SOLVER.CHECKPOINT_PERIOD ${t3_per_epoch} TEST.EVAL_PERIOD ${t3_per_epoch} SOLVER.MAX_ITER ${max_iter} MODEL.YOLO.ARCH models/yolov5${arch}.yaml

epoch=100
((max_iter=epoch*t3_ft_per_epoch))
python main.py --num-gpus ${num_gpu} --config-file training_configs/YOLO-UCD/t3/t3_ft.yaml OUTPUT_DIR YOLO-UCD_${arch}/t3_ft DATALOADER.NUM_WORKERS ${num_workers} MODEL.YOLO.NC 80 MODEL.YOLO.CUR_INTRODUCED_CLS 54 MODEL.WEIGHTS YOLO-UCD_${arch}/t3/model_final.pth MODEL.YOLO.LABEL_SMOOTHING 0.1 SOLVER.IMS_PER_BATCH ${ims_per_batch} SOLVER.CHECKPOINT_PERIOD ${t3_ft_per_epoch} TEST.EVAL_PERIOD ${t3_ft_per_epoch} SOLVER.MAX_ITER ${max_iter} MODEL.YOLO.ARCH models/yolov5${arch}.yaml

epoch=25
((max_iter=epoch*t4_per_epoch))
python main.py --num-gpus ${num_gpu} --config-file training_configs/YOLO-UCD/t4/t4_train.yaml OUTPUT_DIR YOLO-UCD_${arch}/t4 DATALOADER.NUM_WORKERS ${num_workers} MODEL.YOLO.NC 80 MODEL.YOLO.CUR_INTRODUCED_CLS 26 MODEL.WEIGHTS YOLO-UCD_${arch}/t3_ft/model_final.pth MODEL.YOLO.LABEL_SMOOTHING 0.1 SOLVER.IMS_PER_BATCH ${ims_per_batch} SOLVER.CHECKPOINT_PERIOD ${t4_per_epoch} TEST.EVAL_PERIOD ${t4_per_epoch} SOLVER.MAX_ITER ${max_iter} MODEL.YOLO.ARCH models/yolov5${arch}.yaml

epoch=100
((max_iter=epoch*t4_ft_per_epoch))
python main.py --num-gpus ${num_gpu} --config-file training_configs/YOLO-UCD/t4/t4_ft.yaml OUTPUT_DIR YOLO-UCD_${arch}/t4_ft DATALOADER.NUM_WORKERS ${num_workers} MODEL.YOLO.NC 80 MODEL.YOLO.CUR_INTRODUCED_CLS 80 MODEL.WEIGHTS YOLO-UCD_${arch}/t4/model_final.pth MODEL.YOLO.LABEL_SMOOTHING 0.1 SOLVER.IMS_PER_BATCH ${ims_per_batch} SOLVER.CHECKPOINT_PERIOD ${t4_ft_per_epoch} TEST.EVAL_PERIOD ${t4_ft_per_epoch} SOLVER.MAX_ITER ${max_iter} MODEL.YOLO.ARCH models/yolov5${arch}.yaml
