set -x
set -o nounset
set -o errexit

python main.py --num-gpus 1 --dist-url "tcp://127.0.0.1:5778" --config-file training_configs/YOLO-UCD/t4/t4_test.yaml --eval-only MODEL.WEIGHTS YOLO-UCD_s/t4_ft/model_final.pth MODEL.YOLO.ARCH models/yolov5s.yaml OUTPUT_DIR output/YOLO-UCD_s_OSIOD MODEL.YOLO.NC 80 MODEL.YOLO.CALIBRATE 0.2 MODEL.YOLO.TEST_TYPE OSIOD

python main.py --num-gpus 1 --dist-url "cp://127.0.0.1:5778" --config-file training_configs/YOLO-UCD/t4/t4_test.yaml --eval-only MODEL.WEIGHTS YOLO-UCD_s/t4_ft/model_final.pth MODEL.YOLO.ARCH models/yolov5s.yaml OUTPUT_DIR output/YOLO-UCD_s_OSIOD MODEL.YOLO.NC 80 MODEL.YOLO.CALIBRATE 0.2 MODEL.YOLO.TEST_TYPE OSIOD

python main.py --num-gpus 1 --dist-url "tcp://127.0.0.1:5778" --config-file training_configs/YOLO-UCD/t4/t4_test.yaml --eval-only MODEL.WEIGHTS YOLO-UCD_s/t4_ft/model_final.pth MODEL.YOLO.ARCH models/yolov5s.yaml OUTPUT_DIR output/YOLO-UCD_s_OSIOD MODEL.YOLO.NC 80 MODEL.YOLO.CALIBRATE 0.2 MODEL.YOLO.TEST_TYPE OSIOD

python main.py --num-gpus 1 --dist-url "tcp://127.0.0.1:5778" --config-file training_configs/YOLO-UCD/t4/t4_test.yaml --eval-only MODEL.WEIGHTS YOLO-UCD_s/t4_ft/model_final.pth MODEL.YOLO.ARCH models/yolov5s.yaml OUTPUT_DIR output/YOLO-UCD_s_OSIOD MODEL.YOLO.NC 80 MODEL.YOLO.CALIBRATE 0.2 MODEL.YOLO.TEST_TYPE OSIOD