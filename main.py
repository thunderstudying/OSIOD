import logging
import os
from collections import OrderedDict

from copy import deepcopy
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.torch_utils import ModelEMA, de_parallel

import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD, lr_scheduler

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from OSODeval import BasicEvalOperations
from OSIODeval import OSIODEvaluator
import WIC
from detectron2.evaluation import inference_on_dataset
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from training_configs.config import add_yolo_config
from utils.general import intersect_dicts, one_cycle
from models.yolo_d2 import YOLO
from utils.loss import ComputeLoss
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        
        if cfg.MODEL.YOLO.TEST_TYPE == "OSOD":
            evaluator = BasicEvalOperations(dataset_name, cfg, True,
                                            os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name))
        elif cfg.MODEL.YOLO.TEST_TYPE == "OSIOD":
            evaluator = OSIODEvaluator(dataset_name, cfg)
        else:
            raise NotImplementedError("Please set cfg.MODEL.YOLO.TEST_TYPE as OSOD or OSIOD")
    
        results[dataset_name] = inference_on_dataset(model, data_loader, evaluator)

        if cfg.MODEL.YOLO.TEST_TYPE == "OSOD":
            if comm.is_main_process():  # close-set setup
                logger.info(f"Image level evaluation complete for {dataset_name}")
                logger.info(f"Results for {dataset_name}")
                WIC.only_mAP_analysis(results[dataset_name]['predictions']['correct'],
                                    results[dataset_name]['predictions']['scores'],
                                    results[dataset_name]['predictions']['pred_classes'],
                                    results[dataset_name]['category_counts'],
                                    evaluator._coco_api.cats)
                
    if cfg.MODEL.YOLO.TEST_TYPE == "OSOD":  
        if comm.is_main_process():  # open-set setup
            logger.info(f"Combined results for datasets {', '.join(cfg.DATASETS.TEST)}")
            eval_info = {}
            eval_info['category_counts'] = results[list(results.keys())[0]]['category_counts']
            eval_info['predictions'] = {}
            for dataset_name in results:
                for k in results[dataset_name]['predictions'].keys():
                    if k not in eval_info['predictions']:
                        eval_info['predictions'][k] = []
                    eval_info['predictions'][k].extend(results[dataset_name]['predictions'][k])

            WIC.only_mAP_analysis(eval_info['predictions']['correct'],
                                eval_info['predictions']['scores'],
                                eval_info['predictions']['pred_classes'],
                                eval_info['category_counts'])
            Recalls_to_process = (0.1, 0.3, 0.5)
            wilderness = torch.arange(0, 5, 0.1).tolist()
            WIC_values, wilderness_processed = WIC.WIC_analysis(eval_info, Recalls_to_process=Recalls_to_process,
                                                                wilderness=wilderness)
    return


def do_train(cfg, model, resume=False):
    model.train()

    if de_parallel(model)._get_name() == 'YOLO':
        if isinstance(cfg.MODEL.YOLO.HYP, str):
            with open(cfg.MODEL.YOLO.HYP, errors='ignore') as f:
                hyp = yaml.safe_load(f)
        g0, g1, g2 = [], [], []
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                g2.append(v.bias)
            if comm.get_world_size() > 1 and isinstance(v, nn.SyncBatchNorm):
                g0.append(v.weight)
            elif isinstance(v, nn.BatchNorm2d):
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                g1.append(v.weight)
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
        optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
        optimizer.add_param_group({'params': g2})
        del g0, g1, g2

        epochs = cfg.SOLVER.MAX_ITER / cfg.SOLVER.CHECKPOINT_PERIOD
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    else:
        optimizer = build_optimizer(cfg, model)
        scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    if cfg.MODEL.WEIGHTS.endswith('.pth') and de_parallel(model)._get_name() == 'YOLO':
        ckpt = torch.load(cfg.MODEL.WEIGHTS, map_location=f"cuda:{comm.get_local_rank()}")
        exclude = []
        csd = ckpt['model']
        csd = intersect_dicts(csd, de_parallel(model).state_dict(), exclude=exclude)
        de_parallel(model).load_state_dict(csd, strict=False)
        logger.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {cfg.MODEL.WEIGHTS}')

        nc = cfg.MODEL.YOLO.CUR_INTRODUCED_CLS
        imgsz = 640
        nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = cfg.MODEL.YOLO.LABEL_SMOOTHING
        de_parallel(model).nc = nc  # attach number of classes to model
        de_parallel(model).hyp = hyp  # attach hyperparameters to model

        ema = ModelEMA(model)

        if resume is True:
            if ckpt['optimizer'] is not None:
                print('load optimizer')
                optimizer.load_state_dict(ckpt['optimizer'])

            if ckpt['scheduler'] is not None:
                print('load scheduler')
                scheduler.load_state_dict(ckpt['scheduler'])

            if ema and ckpt.get('ema'):
                ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                ema.updates = ckpt['updates']

            start_iter = ckpt['iteration'] + 1

        else:
            start_iter = 0

    elif cfg.MODEL.WEIGHTS == "":
        ema = ModelEMA(model)
        start_iter = 0
        nc = cfg.MODEL.YOLO.CUR_INTRODUCED_CLS
        imgsz = 640
        nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = cfg.MODEL.YOLO.LABEL_SMOOTHING
        de_parallel(model).nc = nc  # attach number of classes to model
        de_parallel(model).hyp = hyp  # attach hyperparameters to model

    else:
        start_iter = (
                checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
        )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter, max_to_keep=10
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    if de_parallel(model)._get_name() == 'YOLO':
        compute_loss = ComputeLoss(de_parallel(model))

    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            if de_parallel(model)._get_name() == 'YOLO':
                imgs = []
                targets = []
                for i, x in enumerate(data):
                    imgs.append(x['YOLO_image'][None])
                    x['targets'][:, 0] = i
                    targets.append(x['targets'])
                imgs = torch.vstack(imgs).to(f"cuda:{comm.get_local_rank()}").float() / 255
                targets = torch.vstack(targets)
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(f"cuda:{comm.get_local_rank()}"))
                loss_dict = {"lbox": loss_items[0], "lobj": loss_items[1], "lcls": loss_items[2]}
                losses = loss * comm.get_world_size()

            else:
                loss_dict = model(data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if de_parallel(model)._get_name() == 'YOLO':
                ema.update(model)

            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            if de_parallel(model)._get_name() != 'YOLO':
                scheduler.step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter
            ):
                if de_parallel(model)._get_name() == 'YOLO':
                    scheduler.step()
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()

            if de_parallel(model)._get_name() == 'YOLO':
                periodic_checkpointer.step(iteration, ema=deepcopy(ema.ema), updates=ema.updates)
            else:
                periodic_checkpointer.step(iteration)


def setup(args):
    cfg = get_cfg()
    add_yolo_config(cfg)  # config.py
    cfg.merge_from_file(args.config_file)  # yaml
    cfg.merge_from_list(args.opts)  # command line
    cfg.freeze()
    default_setup(
        cfg, args
    )
    register_coco_instances('t1_coco_2017_train',
                            {},
                            f'protocol/custom_protocols/OSIOD_t1.json',
                            f'datasets/COCO2017/JPEGImages')
    register_coco_instances('t2_coco_2017_train',
                            {},
                            f'protocol/custom_protocols/OSIOD_t2.json',
                            f'datasets/COCO2017/JPEGImages')
    register_coco_instances('t2_coco_2017_ft',
                            {},
                            f'protocol/custom_protocols/OSIOD_t2_ft.json',
                            f'datasets/COCO2017/JPEGImages')
    register_coco_instances('t3_coco_2017_train',
                            {},
                            f'protocol/custom_protocols/OSIOD_t3.json',
                            f'datasets/COCO2017/JPEGImages')
    register_coco_instances('t3_coco_2017_ft',
                            {},
                            f'protocol/custom_protocols/OSIOD_t3_ft.json',
                            f'datasets/COCO2017/JPEGImages')
    register_coco_instances('t4_coco_2017_train',
                            {},
                            f'protocol/custom_protocols/OSIOD_t4.json',
                            f'datasets/COCO2017/JPEGImages')
    register_coco_instances('t4_coco_2017_ft',
                            {},
                            f'protocol/custom_protocols/OSIOD_t4_ft.json',
                            f'datasets/COCO2017/JPEGImages')
    for protocol_name in (
            'no_difficult_custom_voc_2007_train', 'no_difficult_custom_voc_2007_val',
            'no_difficult_custom_voc_2012_train', 'no_difficult_custom_voc_2012_val',
            'no_difficult_voc_2007_test'):
        register_coco_instances(protocol_name,
                                {},
                                f"protocol/custom_protocols/{protocol_name}.json",
                                f"datasets/VOC{protocol_name.split('_')[-2]}/JPEGImages/")
    register_coco_instances("WR1_Mixed_Unknowns", {}, "protocol/custom_protocols/WR1_Mixed_Unknowns.json",
                            "datasets/COCO2017/JPEGImages")
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        if cfg.MODEL.META_ARCHITECTURE == 'YOLO':
            weights = cfg.MODEL.WEIGHTS
            if weights.endswith('.pt'):  # trained from YOLOv5
                ckpt = torch.load(weights)
                exclude = ['anchor'] if (cfg.MODEL.YOLO.ARCH or cfg.MODEL.YOLO.ANCHORS) and not args.resume else []
                csd = ckpt['model'].float().state_dict()
                csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
                model.load_state_dict(csd, strict=False)

            else:  # trained from YOLO-UCD
                ckpt = torch.load(weights)
                exclude = []
                csd = ckpt['model']
                csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
                model.load_state_dict(csd, strict=True)

            logger.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
