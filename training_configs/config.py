from detectron2.config import CfgNode as CN


def add_yolo_config(cfg):
    cfg.MODEL.YOLO = CN()
    cfg.MODEL.YOLO.ARCH = None  # 'models/yolov5s.yaml'
    cfg.MODEL.YOLO.CH = 3
    cfg.MODEL.YOLO.NC = 20
    cfg.MODEL.YOLO.AGNOSTIC = False
    cfg.MODEL.YOLO.LABEL_SMOOTHING = 0.0
    cfg.MODEL.YOLO.CALIBRATE = None
    cfg.MODEL.YOLO.CUR_INTRODUCED_CLS = None
    cfg.MODEL.YOLO.ANCHORS = None

    cfg.MODEL.YOLO.CONF_THRES_TEST = 0.05
    cfg.MODEL.YOLO.IOU_THRES_TEST = 0.5

    cfg.MODEL.YOLO.MAX_DET_TEST = 300
    cfg.MODEL.YOLO.MULTI_LABEL_TEST = True

    cfg.MODEL.YOLO.HYP = 'data/hyps/hyp.scratch.yaml'
    cfg.MODEL.YOLO.IMGSZ = 640

    cfg.MODEL.YOLO.TEST_TYPE = "OSOD"  # OSOD OSIOD
