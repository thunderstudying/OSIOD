# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))  # 输入经过sigmoid后再进行BCELoss
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # print('hyp_loss:', h)
        print('label_smoothing:', h.get('label_smoothing', 0.0))
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj  置信度

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets  [batch_size,anchor_num,heights,width,prediction(class+xywh)]->[filtered_nt,xywh+score+classes]

                # Regression  只计算正样本边界框的loss
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness 置信度 计算所有（正负）样本的loss
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio self.gr越大越接近iou self.gr越小置信度越接近1（人为加大训练难度）

                # Classification 只计算正样本的loss
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp  # [filtered_nt, classes_num]
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE 训练的时候类概率没有使用sigmoid？使用了，在损失函数计算的时候激活 99行 BCEcls

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)  # tobj每个位置都参与loss的计算 有目标的score的gt为iou没有目标的gt为0
            # 每个feature map的置信度损失权重不同  要乘以相应的权重系数self.balance[i]
            # 一般来说，检测小物体的难度大一点，所以会增加大特征图的损失系数，让模型更加侧重小物体的检测
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # p
        # [batch_size, anchor_num, grid_h, grid_w, xywh+class+classes]
        # [32, 3, 112, 112, 85], [32, 3, 56, 56, 85], [32, 3, 28, 28, 85]
        # targets [target_num, image_ind+class+xywh]
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain  image_ind+class+xywh+anchor_index
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # ai 标记目标属于哪个anchor [1,3]->[3,1]->[3,target_num]=[na,nt] 第一个行target_num个0 二行..个1 三行..个2
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        # targets复制3份对应一个特征图的3个anchor
        # [target_num, 6][3, target_num]->[3, target_num, 6][3, target_num, 1]->[3, target_num, 7]
        # [3, target_num, img_ind+class+xywh+属于哪个anchor的标记]
        g = 0.5  # bias  衡量target离哪个格子更近
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        # 自身 + 周围上下左右 计算offsets

        for i in range(self.nl):  # 每个 Detect
            anchors = self.anchors[i]  # 当前特征图对应三个anchor的尺寸 [3, 3, 2]->[3, 2]

            # 保存每个输出特征图的宽高
            # [1, 1, 1, 1, 1, 1, 1] ->[1, 1, 112, 112, 112, 112, 1] = image_ind+class+xywh+anchor_index
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # 将归一化尺度放缩到当前特征图的坐标尺度
            # [3, target_num, 7] -> [3, target_num, image_ind+class+xywh+anchor_ind]
            t = targets * gain
            if nt:
                # Matches
                # t=[na, nt, 7] t[:, :, 4:6]=[na, nt, 2] anchors=[na, 2] anchors[: , None]=[na, 1, 2]
                # r=[na, nt, 2] gt与三个anchors的宽高比
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # [na, nt] True gt是当前anchor的正样本 False gt是当前anchor的负样本
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))  # yolov3,4的筛选方法
                t = t[j]  # filter 过滤负样本，得到当前特征图上三个anchor的所有正样本 [positive_target(filtered_nt), image_ind+class+xywh+anchor_ind]

                # Offsets筛选当前格子周围的格子，找到离gt中心最近的两个格子，三个格子计算loss
                gxy = t[:, 2:4]  # grid xy 标注是中心点xywh t[:, 2:4]得到中心点的xy（相对于左上角）
                gxi = gain[[2, 3]] - gxy  # inverse  中心点相对于右下角的坐标
                # gxy>1 gxi>1 表示中心点不在边上，在边上就没有上下左右四个格子了
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  # j bool 左边格子是否计算loss k bool 上面格子是否计算loss [filtered_nt]
                l, m = ((gxi % 1 < g) & (gxi > 1)).T  # l bool 右边格子是否计算loss m bool 下面格子是否计算loss [filtered_nt]
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # [5, filtered_nt] 当前格子 左边格子 上面格子 右边格子 下面格子
                t = t.repeat((5, 1, 1))[j]  # t.repeat((5,1,1)) [filtered_nt]->[5, filtered_nt, 7] [5, filtered_nt, 7][5, filtered_nt]=[filtered_nt*3, 7] 当没有边界的格子时为等号 有边界格子时为小于号
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                # torch.zeros_like(gxy)[None] [1, filtered_nt, 2] + off[:, None] [5, 1, 2] -> [5,filtered_nt,2]
                # [5,filtered_nt,2][5,filtered_nt] -> [filtered*3,2]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()  # 负责预测的网格的左上角坐标
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch  # 分类GT 边界框GT 负责预测的anchor和grid索引 anchor值
