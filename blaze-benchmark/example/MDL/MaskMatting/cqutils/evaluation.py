import numpy as np
import cv2
import traceback
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
import math
from collections import Counter
import sys

__all__ = [
        'AverageMeter',
        'seg_eval',
        'SaliencyEvaluation',
        'SegmentationEvaluation',
        'MattingEvaluation',
        'DetectionEvaluation',
        ]

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seg_eval(predictions, gts, num_clses):
    """
    segmentation evaluation
    """
    hist = np.zeros((num_clses, num_clses))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_clses)
    # axis 0: gt, axis 1: predition
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist)/hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq>0]*iu[freq>0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def _fast_hist(label_p, label_t, num_clses):
    mask = (label_t>=0) & (label_t<num_clses)
    hist = np.bincount(
            num_clses*label_t[mask].astype(int) + 
            label_p[mask], minlength=num_clses ** 2).reshape(num_clses,
                    num_clses)
    return hist

class SaliencyEvaluation(object):
    def __init__(self):
        self.max_threshold = 255
        self.epsilon = 1e-8
        self.beta = 0.3
        self.weighted_f = WeightedF()
        self.sm = SM()
        self.em = EM()

        self.MAE = 0
        self.Precision = np.zeros(256)
        self.Recall = np.zeros(256)
        self.S_Measure = 0
        self.E_Measure = 0
        self.WFb = 0
        self.num = 0.0

    def add_one(self, predict, gt):
        try:
            MAE, Precision, Recall, _, S_Measure, WFb, E_Measure\
                = self.evaluation(predict, gt)
            self.MAE += MAE
            self.Precision += Precision
            self.Recall += Recall
            self.S_Measure += S_Measure
            self.E_Measure += E_Measure
            self.WFb += WFb
            self.num += 1.0
        except:
            print(traceback.print_exc())

    def clear(self):
        self.MAE = 0
        self.Precision = np.zeros(256)
        self.Recall = np.zeros(256)
        self.S_Measure = 0
        self.E_Measure = 0
        self.WFb = 0
        self.num = 0

    def get_evaluation(self):
        if self.num > 0:
            avg_MAE = self.MAE / self.num
            avg_Precision = self.Precision / self.num
            avg_Recall = self.Recall / self.num
            avg_S_Measure = self.S_Measure / self.num
            avg_E_Measure = self.E_Measure / self.num
            avg_WFb = self.WFb / self.num
            F_m = (1.3 * avg_Precision * avg_Recall) / (0.3 * avg_Precision + avg_Recall)
            return avg_MAE, avg_Precision, avg_Recall,\
                   F_m, avg_S_Measure, avg_WFb, avg_E_Measure
        else:
            return 0, np.zeros(256), np.zeros(256), np.zeros(256), 0, 0, 0

    def evaluation(self, predict, gt):
        MAE = self.mae(predict, gt)
        # gt[gt > 25] = 255
        # gt[gt <= 25] = 0
        Precision, Recall = self.precision_and_recall(predict, gt)
        # F_measure = self.f_measure(Precision, Recall)
        # w_f = self.weighted_f.get_weighted_F(predict/255.0, gt/255.0)
        S_Measure = self.sm.eval(predict, gt)
        E_Measure = self.em.eval(predict, gt)
        WFb = self.weighted_f.eval(predict, gt)
        return MAE, Precision, Recall, 0, S_Measure, WFb, E_Measure

    def best_evaluation(self, predict, gt):
        MAE = self.mae(predict, gt)
        Precision, Recall = self.precision_and_recall(predict, gt)
        F_measure = self.f_measure(Precision, Recall)
        idx = np.argmax(F_measure)
        return MAE, Precision[idx], Recall[idx], F_measure[idx]

    def adaptive_evaluation(self, predict, gt):
        MAE = self.mae(predict, gt)
        threshold = 2 * np.mean(predict)
        threshold = 254.0 if threshold > 254.0 else threshold
        Precision, Recall = self.precision_and_recall(predict, gt,
                                                      threshold=threshold)
        F_measure = self.f_measure(Precision, Recall)
        return MAE, Precision, Recall, F_measure

    def mae(self, predict, gt):
        '''
            predict: numpy, shape is (height, width), value 0-255
            gt: numpy, shape is (height, width), value 0-255
        '''
        assert predict.shape == gt.shape
        return np.mean(np.abs(predict - gt)/255.0)

    def f_measure(self, precision, recall):
        f = ((1 + self.beta) * precision * recall) / (self.beta * precision + recall)
        return f

    def precision_and_recall(self, predict, gt, threshold=None):
        '''
            predict: numpy, shape is (height, width), value 0-255
            gt: numpy, shape is (height, width), value 0-255
        '''
        assert predict.shape == gt.shape
        pred_max = np.max(predict)
        if pred_max > 0:
            predict[predict < 0] = 0
            predict = predict.astype(np.float)
            predict *= 255.0/pred_max
            predict = np.round(predict).astype(np.int)

        if threshold is None:
            if pred_max > 0:
                GT = np.zeros(gt.shape, dtype=np.int)
                GT[gt > 0] = 1
                predict_th = np.cumsum(np.bincount(
                    predict.flatten(), minlength=256)[::-1])
                predict_precision_th = np.cumsum(np.bincount(
                    predict[GT==1].flatten(), minlength=256)[::-1])
                precision = predict_precision_th / predict_th.astype(np.float)
                recall = predict_precision_th/ float(np.sum(GT))
                return precision[::-1], recall[::-1]
            else:
                return np.zeros(256), np.zeros(256)

            # precision = []
            # recall = []
            # for th in range(self.max_threshold):
            #     # ground true pixel
            #     TS = np.zeros(predict.shape)
            #     # predicted true pixel
            #     DS = np.zeros(predict.shape)
            #
            #     TS[gt > (self.max_threshold / 2)] = 1
            #     DS[predict > th] = 1
            #     TSDS = TS * DS
            #
            #     precision.append((np.mean(TSDS)+self.epsilon) / (np.mean(DS)+self.epsilon))
            #     recall.append((np.mean(TSDS)+self.epsilon) / (np.mean(TS)+self.epsilon))
            # return np.array(precision), np.array(recall)
        else:
            # ground true pixel
            TS = np.zeros(predict.shape)
            # predicted true pixel
            DS = np.zeros(predict.shape)

            TS[gt > (self.max_threshold / 2)] = 1
            DS[predict > threshold] = 1
            TSDS = TS * DS

            precision = (np.mean(TSDS)+self.epsilon) / (np.mean(DS)+self.epsilon)
            recall = (np.mean(TSDS)+self.epsilon) / (np.mean(TS)+self.epsilon)
            return precision, recall

    def get_mask(self, gt):
        mask = np.zeros(gt.shape)
        _, contours, _ = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for item in contours:
            w_min, w_max = np.min(item[:, 0, 0]), np.max(item[:, 0, 0])
            h_min, h_max = np.min(item[:, 0, 1]), np.max(item[:, 0, 1])
            mask[h_min:h_max+1, w_min:w_max+1] = 1
        return mask

class SegmentationEvaluation(object):
    def __init__(self):
        self.mpa = 0
        self.mIoU = 0
        self.num = 0
        self.max_threshold = 255
        self.epsilon = 1e-8
        self.IoU_list  = []

    def add_one(self, predict, gt, th=125):
        try:
            mpa = self.get_pa(predict, gt, th)
            mIoU = self.get_mIoU(predict, gt, th)
            self.mpa += mpa
            self.mIoU += mIoU
            self.IoU_list.append(mIoU)
            self.num += 1
        except:
            print(traceback.print_exc())

    def evaluation(self):
        if self.num > 0:
            return self.mpa/self.num, self.mIoU/self.num
        else:
            return 0, 0

    def get_IoU_list(self):
        return self.IoU_list

    def clear(self):
        self.mpa = 0
        self.mIoU = 0
        self.num = 0
        self.IoU_list = []

    def get_pa(self, predict, gt, th=125):
        GT = np.zeros(predict.shape)
        P = np.zeros(predict.shape)

        GT[gt > (self.max_threshold // 2)] = 1
        P[predict > th] = 1
        Precision_F = np.sum(P * GT) / (np.sum(P) + self.epsilon)
        Precision_B = np.sum((1 - P) * (1 - GT)) / (np.sum(1-P) + self.epsilon)

        mpa = (Precision_F+Precision_B)/2
        return mpa

    def get_mIoU(self, predict, gt, th=125):
        GT = np.zeros(predict.shape)
        P = np.zeros(predict.shape)

        GT[gt > (self.max_threshold // 2)] = 1
        P[predict > th] = 1

        # foreground
        mpa = np.sum(P * GT) / (np.sum(P)+np.sum(GT)-np.sum(P * GT)+self.epsilon)
        # background
        mpa += np.sum((1 - P) * (1 - GT)) /\
               (np.sum(1-P)+np.sum(1-GT)-np.sum((1 - P) * (1 - GT))+self.epsilon)
        return mpa/2

class MattingEvaluation(object):
    def __init__(self):
        self.MAE = 0
        self.SAD = 0
        self.MSE = 0
        self.GRAD = 0
        self.CONN = 0
        self.mPA = 0
        self.mIoU = 0
        self.num = 0
        self.max_threshold = 255
        self.epsilon = 1e-8

    def add_one(self, predict, gt, th=125):
        try:
            tmp_MAE = self.mae(predict, gt)
            tmp_SAD = self.sad(predict, gt)
            tmp_MSE = self.mse(predict, gt)
            tmp_GRAD = self.grad(predict, gt)
            # tmp_CONN = self.connectivity(predict, gt)
            tmp_mPA = self.get_pa(predict, gt, th)
            tmp_mIoU = self.get_mIoU(predict, gt, th)
            self.MAE += tmp_MAE
            self.SAD += tmp_SAD
            self.MSE += tmp_MSE
            self.GRAD += tmp_GRAD
            # self.CONN += tmp_CONN
            self.mPA += tmp_mPA
            self.mIoU += tmp_mIoU
            self.num += 1
        except:
            print(traceback.print_exc())

    def evaluation(self):
        if self.num > 0:
            return self.MAE / self.num, self.SAD / self.num, \
                   self.MSE / self.num, self.GRAD / self.num, \
                   self.CONN / self.num, self.mPA/ self.num, \
                   self.mIoU / self.num
        else:
            return 0, 0, 0, 0, 0, 0, 0

    def clear(self):
        self.MAE = 0
        self.SAD = 0
        self.MSE = 0
        self.GRAD = 0
        self.CONN = 0
        self.mPA = 0
        self.mIoU = 0
        self.num = 0

    def mae(self, predict, gt):
        '''
            predict: numpy, shape is (height, width), value 0-255
            gt: numpy, shape is (height, width), value 0-255
        '''
        assert predict.shape == gt.shape
        pred_norm = predict / 255.0
        gt_norm = gt / 255.0
        return np.mean(np.abs(pred_norm - gt_norm))

    def mse(self, pred, gt):
        assert (pred.shape == gt.shape)
        assert (len(pred.shape) == 2)
        num = pred.shape[0] * pred.shape[1]
        pred_norm = pred / 255.0
        gt_norm = gt / 255.0
        mse = np.mean((pred_norm - gt_norm) ** 2)
        return mse

    def sad(self, predict, gt):
        assert predict.shape == gt.shape
        pred_norm = predict / 255.0
        gt_norm = gt / 255.0
        return np.sum(np.abs(pred_norm - gt_norm))

    def grad(self, predict, gt):
        assert (predict.shape == gt.shape)
        assert (len(predict.shape) == 2)
        num = predict.shape[0] * predict.shape[1]

        pred_norm = predict / 255.0
        gt_norm = gt / 255.0

        grad_pred = self.gauss_gradient(pred_norm, 1.4)
        grad_gt = self.gauss_gradient(gt_norm, 1.4)
        err = np.mean((grad_pred - grad_gt) ** 2)
        return err

    def gauss_gradient(self, I, sigma):
        epsilon = 0.02
        halfsize = sigma * np.sqrt(-2 * np.log(epsilon * np.sqrt(2 * np.pi) *
                                               sigma))
        halfsize = np.ceil(halfsize)

        t = np.arange(-halfsize, halfsize + 1, dtype='float')
        k_u = np.exp(-t * t / (2 * sigma * sigma))
        k_u /= k_u.sum()
        k_v = -k_u * t / (sigma * sigma)

        I = I.astype('float')
        dx = ndimage.convolve1d(I, k_u, axis=0, mode='nearest')
        dx = ndimage.convolve1d(dx, k_v, axis=1, mode='nearest')

        dy = ndimage.convolve1d(I, k_u, axis=1, mode='nearest')
        dy = ndimage.convolve1d(dy, k_v, axis=0, mode='nearest')

        return np.sqrt(dx * dx + dy * dy)

    def connectivity(self, alpha, gt):
        num = alpha.shape[0] * alpha.shape[1]
        I = (alpha >= 254) * (gt == 255)
        labeled, nr_objects = ndimage.label(I > 0.5)
        # print('nr_objects:{}'.format(nr_objects))

        size = 0
        omega = np.zeros_like(labeled)
        for i in range(1, nr_objects + 1):
            mask = (labeled == i)
            s = mask.sum()
            # print s
            if s > size:
                size = s
                omega = mask

        phi0 = self.compute_phi(alpha, omega)
        phi1 = self.compute_phi(gt, omega)
        # cv2.imwrite('phi0.png', (phi0 * 255).astype('uint8'))
        # cv2.imwrite('phi1.png', (phi1 * 255).astype('uint8'))
        conn = np.sum(np.abs(phi0 - phi1))
        conn_pixel = conn / float(num)
        return conn_pixel

    def compute_phi(self, alpha, omega):
        mask = np.ones(omega.shape)
        Li = np.ones(omega.shape) * 255
        for v in range(255):
            I = (alpha > v) * mask
            if np.sum(I) == 0:
                break

            labeled, nr_c = ndimage.label(I > 0.5)
            for i in range(nr_c):
                cur_mask = (labeled == i + 1)
                if np.any(cur_mask * omega):
                    # omega component
                    new_mask = (labeled != i + 1)
                    Li[np.logical_and(new_mask, mask)] = v
                    mask = cur_mask
                    break
        di = (alpha - Li) / 255.
        di[di < 0.15] = 0.
        phi = 1 - di
        return phi

    def get_pa(self, predict, gt, th=125):
        GT = np.zeros(predict.shape)
        P = np.zeros(predict.shape)

        GT[gt > th] = 1
        P[predict > th] = 1
        Precision_F = np.sum(P * GT) / (np.sum(P) + self.epsilon)
        Precision_B = np.sum((1 - P) * (1 - GT)) / (np.sum(1-P) + self.epsilon)

        mpa = (Precision_F+Precision_B)/2
        return mpa

    def get_mIoU(self, predict, gt, th=125):
        GT = np.zeros(predict.shape)
        P = np.zeros(predict.shape)

        GT[gt > th] = 1
        P[predict > th] = 1

        # foreground
        mpa = np.sum(P * GT) / (np.sum(P)+np.sum(GT)-np.sum(P * GT)+self.epsilon)
        # background
        mpa += np.sum((1 - P) * (1 - GT)) /\
               (np.sum(1-P)+np.sum(1-GT)-np.sum((1 - P) * (1 - GT))+self.epsilon)
        return mpa/2

class SM(object):
    def __init__(self):
        self.alpha = 0.5
        self.eps = 1e-8

    def eval(self, predict, gt):
        dGT = gt.astype(np.float) / 255.0
        y = dGT.mean()

        if y == 0:
            x = (predict / 255.0).mean()
            Q = 1.0 - x
        elif y == 1:
            x = (predict / 255.0).mean()
            Q = x
        else:
            alpha = 0.5
            Q = alpha * self.so(predict, gt) \
                + (1 - alpha) * self.sr(predict, gt)
            if Q < 0:
                Q = 0

        return Q

    def so(self, predict, gt):
        prediction_fg = predict.copy() / 255.0
        prediction_fg[gt == 0] = 0
        O_FG = self.object(prediction_fg, gt)

        prediction_bg = 1.0 - predict / 255.0
        prediction_bg[gt > 0] = 0
        O_BG = self.object(prediction_bg, gt == 0)

        u = (gt / 255.0).mean()
        Q = u * O_FG + (1 - u) * O_BG

        return Q

    def object(self, predict, gt):
        x = (predict[gt > 0]).mean()
        sigma_x = (predict[gt > 0]).std()

        score = 2.0 * x / (x**2 + 1.0 + sigma_x + self.eps)

        return score

    def sr(self, predict, gt):
        X, Y = self.centroid(gt)

        GT_1, GT_2, GT_3, GT_4, w1, w2, w3, w4 \
            = self.divideGT(gt, X, Y)

        prediction_1, prediction_2, prediction_3, prediction_4 \
            = self.Divideprediction(predict, X, Y)

        Q1 = self.ssim(prediction_1, GT_1)
        Q2 = self.ssim(prediction_2, GT_2)
        Q3 = self.ssim(prediction_3, GT_3)
        Q4 = self.ssim(prediction_4, GT_4)

        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        return Q

    def ssim(self, predict, gt):
        hei, wid = predict.shape[:2]
        N = wid * hei
        prediction = predict.copy()
        dGT = gt.astype(np.float)

        x = (prediction).mean()
        y = (dGT).mean()

        # sigma_x2 = var(prediction(:))
        sigma_x2 = ((prediction - x)**2).sum() / (N - 1 + self.eps)
        # sigma_y2 = var(dGT(:))
        sigma_y2 = ((dGT - y)**2).sum() / (N - 1 + self.eps)

        sigma_xy = ((prediction - x) * (dGT - y)).sum() / (N - 1 + self.eps)

        alpha = 4 * x * y * sigma_xy
        beta = (x**2 + y**2) * (sigma_x2 + sigma_y2)

        if alpha != 0:
            Q = alpha / (beta + self.eps)
        elif alpha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

    def Divideprediction(self, predict, X, Y):
        hei, wid = predict.shape[:2]

        LT = predict[:Y, :X].copy() / 255.0
        RT = predict[:Y, X:wid].copy() / 255.0
        LB = predict[Y:hei, :X].copy() / 255.0
        RB = predict[Y:hei, X:wid].copy() / 255.0

        return LT, RT, LB, RB

    def divideGT(self, gt, X, Y):
        hei, wid = gt.shape[:2]
        area = wid * hei

        LT = gt[:Y, :X].copy() / 255.0
        RT = gt[:Y, X: wid].copy() / 255.0
        LB = gt[Y:hei, :X].copy() / 255.0
        RB = gt[Y:hei, X:wid].copy() / 255.0

        w1 = (X * Y) / area
        w2 = ((wid - X) * Y) / area
        w3 = (X * (hei - Y)) / area
        w4 = 1.0 - w1 - w2 - w3

        return LT, RT, LB, RB, w1, w2, w3, w4

    def centroid(self, gt):
        rows, cols = gt.shape[:2]
        dGT = gt.astype(np.float) / 255.0

        if gt.sum() == 0:
            X = round(cols / 2)
            Y = round(rows / 2)
        else:
            total = (gt.astype(np.float) / 255.0).sum()
            i = np.array(range(cols))
            j = np.array(range(rows))
            X = int(round((dGT.sum(0) * i).sum() / total))
            Y = int(round((dGT.sum(1) * j).sum() / total))

        return X, Y

class EM(object):
    def __init__(self):
        self.eps = 1e-8

    def eval(self, predict, gt):
        '''
        @conference{Fan2018Enhanced, title={Enhanced-alignment Measure for Binary Foreground Map Evaluation},
                    author={Fan, Deng-Ping and Gong, Cheng and Cao, Yang and Ren, Bo and Cheng, Ming-Ming and Borji, Ali},
                    year = {2018},
                    booktitle = {IJCAI}
        }
        Input:
            predict - Binary/Non binary foreground map with values in the range [0 255]. Type: double.
            gt - Binary/Non binary foreground map with values in the range [0 255]. Type: double, uint8.
        Output:
            score - The score
        '''
        dGT = gt.astype(np.float) / 255.0
        dFG = predict.astype(np.float) / 255.0

        if dGT.sum() == 0:
            enhanced_matrix = 1.0 - dFG
        elif (dGT == 0).sum() == 0:
            enhanced_matrix = dFG
        else:
            align_matrix = self.AlignmentTerm(dFG, dGT)
            enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)

        h, w = gt.shape[:2]
        score = enhanced_matrix.sum()/ (w * h - 1 + self.eps)

        return score

    def AlignmentTerm(self, dFM, dGT):
        mu_FM = dFM.mean()
        mu_GT = dGT.mean()

        align_FM = dFM - mu_FM
        align_GT = dGT - mu_GT

        align_Matrix = 2. * (align_GT * align_FM) / (align_GT * align_GT + align_FM * align_FM + self.eps)
        return align_Matrix

    def EnhancedAlignmentTerm(self, align_Matrix):
        enhanced = ((align_Matrix + 1)**2) / 4
        return enhanced


class WeightedF(object):
    def __init__(self):
        self.eps = 1e-8

    def eval(self, predict, gt):
        '''
        WFb Compute the Weighted F-beta measure (as proposed in "How to Evaluate
        Foreground Maps?" [Margolin et. al - CVPR'14])
        Input:
            predict - Binary/Non binary foreground map with values in the range [0 255]. Type: double.
            gt - Binary/Non binary foreground map with values in the range [0 255]. Type: double, uint8.
        Output:
            Q - The Weighted F-beta score
        '''
        dGT = gt.astype(np.float) / 255.0
        FG = predict.astype(np.float) / 255.0

        E = np.abs(FG - dGT)
        # [Ef, Et, Er] = deal(abs(FG - GT));

        # [Dst, IDXT] = bwdist(dGT)
        # IDXT: (idx0, idx1)
        Dst, IDXT = ndimage.distance_transform_edt(dGT == 0, return_indices=True)


        Et = E.copy()
        IDX_h = IDXT[0]
        IDX_w = IDXT[1]
        Et[gt==0] = Et[IDX_h[gt==0], IDX_w[gt==0]]

        # K = fspecial('gaussian', 7, 5)
        # EA = imfilter(Et, K)
        EA = gaussian_filter(Et, sigma=5, truncate=7)
        MIN_E_EA = E.copy()
        mask = (gt > 0) * (EA < E)
        MIN_E_EA[mask] = EA[mask]

        B = np.ones_like(gt)
        B[gt==0] = 2 - 1 * np.exp(math.log(1 - 0.5) / 5 * Dst[gt==0])
        Ew = MIN_E_EA * B

        TPw = dGT.sum() - Ew[gt>0].sum()
        FPw = Ew[gt==0].sum()

        R = 1 - Ew[gt>0].mean()
        P = TPw / (self.eps + TPw + FPw)

        Q = (2) * (R * P) / (self.eps + R + P) # Beta = 1
        # Q = (1 + Beta ^ 2) * (R * P). / (eps + R + (Beta. * P));
        return Q

class DetectionEvaluation(object):
    def __init__(self, classes=2):
        self.classes = classes
        self.num = 0
        self.detections = []
        self.groundTruths = []
        self.IOUThreshold = 0.5

    def clear(self):
        self.num = 0
        self.detections = []
        self.groundTruths = []

    def add_one(self, pred, gt):
        # N, x1, y1, x2, y2, obj_conf, class_conf, class_pred
        for item in pred:
            # [imageName, class, confidence, (bb coordinates XYX2Y2)]
            self.detections.append([
                str(self.num),
                int(item[6]),
                item[4],
                item[:4]
            ])
        # N, class_id, cx, cy, w, h
        for item in gt:
            # [imageName, class, confidence, (bb coordinates XYX2Y2)]
            self.groundTruths.append([
                str(self.num),
                int(item[0]),
                1,
                [item[1] - item[3] // 2,
                 item[2] - item[4] // 2,
                 item[1] + item[3] // 2,
                 item[2] + item[4] // 2]
            ])
        self.num += 1

    def eval(self):
        """
        Returns:
            mAP: mean average precision;
            cls_acc: classification accuracy for every class
        """
        if self.num == 0:
            return 0, 0

        cls_acc = {}
        mpa = 0
        ap, mIoU = self.get_ap_miou()
        for item in ap:
           mpa += item['AP']
           cls_acc[item['class']] = item['total_TP']
        mpa = float(mpa) / float(self.classes)

        return mpa, cls_acc, mIoU

    def person_object_pre(self):
        """person and object classification precision
        Returns:
            pre: list [person_pre, object_pre];
        """

        # gt person set
        gts = []
        [gts.append(g[0]) for g in self.groundTruths if g[1] == 1]
        gt_set = set(gts)
        # pred person set
        pred = []
        [pred.append(p[0]) for p in self.detections if p[1] == 1]
        pred_set = set(pred)
        merge_person = gt_set & pred_set
        if len(gt_set) == 0:
            person_pre = 0
        else:
            person_pre = len(merge_person) / len(gt_set)

        # gt object set
        gts = []
        [gts.append(g[0]) for g in self.groundTruths if g[1] == 0]
        gt_set = set(gts)
        # pred object set
        pred = []
        [pred.append(p[0]) for p in self.detections if p[1] == 0]
        pred_set = set(pred)
        merge_object = gt_set & pred_set
        if len(gt_set) == 0:
            object_pre = 0
        else:
            object_pre = len(merge_object) / len(gt_set)

        return [person_pre, object_pre]

    def get_ap_miou(self):
        """Get the metrics used by the VOC Pascal 2012 challenge.
        Returns:
            AP: average precision;
        """
        ret = []  # list containing metrics (precision, recall, average precision) of each class
        mIoU = 0
        # Precision x Recall is obtained individually by each class
        # Loop through by classes
        for c in range(self.classes):
            # Get only detection of class c
            dects = []
            [dects.append(d) for d in self.detections if d[1] == c]
            # Get only ground truths of class c
            gts = []
            [gts.append(g) for g in self.groundTruths if g[1] == c]
            npos = len(gts)
            # sort detections by decreasing confidence
            dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
            # create dictionary with amount of gts for each image
            det = Counter([cc[0] for cc in gts])
            for key, val in det.items():
                det[key] = np.zeros(val)
            # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
            # Loop through detections
            for d in range(len(dects)):
                # print('dect %s => %s' % (dects[d][0], dects[d][3],))
                # Find ground truth image
                gt = [gt for gt in gts if gt[0] == dects[d][0]]
                iouMax = 0
                for j in range(len(gt)):
                    # print('Ground truth gt => %s' % (gt[j][3],))
                    iou = self.iou(dects[d][3], gt[j][3])
                    if iou > iouMax:
                        iouMax = iou
                        jmax = j
                # Assign detection as true positive/don't care/false positive
                mIoU += iouMax
                if iouMax >= self.IOUThreshold:
                    if det[dects[d][0]][jmax] == 0:
                        TP[d] = 1  # count as true positive
                        det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                        # print("TP")
                    else:
                        FP[d] = 1  # count as false positive
                        # print("FP")
                # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
                else:
                    FP[d] = 1  # count as false positive
                    # print("FP")

            mIoU /= float(len(dects))
            # compute precision, recall and average precision
            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))
            # Depending on the method, call the right implementation
            [ap, mpre, mrec, ii] = self.CalculateAveragePrecision(rec, prec)
            # add class result in the dictionary to be returned
            r = {
                'class': c,
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total_positives': npos,
                'total_TP': np.sum(TP),
                'total_FP': np.sum(FP)
            }
            ret.append(r)
        return ret, mIoU

    def CalculateAveragePrecision(self, rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    # 11-point interpolated average precision
    def ElevenPointInterpolatedAP(self, rec, prec):
        # def CalculateAveragePrecision2(rec, prec):
        mrec = []
        # mrec.append(0)
        [mrec.append(e) for e in rec]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e) for e in prec]
        # mpre.append(0)
        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rhoInterp) / 11
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)
        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]
        return [ap, rhoInterp, recallValues, None]

    def iou(self, boxA, boxB):
        """
        Inputs:
            boxA: list [x1, y1, x2, y2];
            boxB: list [x1, y1, x2, y2];
        """

        intersection_area = (min(boxA[1], boxB[1]) - max(boxA[0], boxB[0])) * \
                            (min(boxA[3], boxB[3]) - max(boxA[2], boxB[2]))

        merge_area = (max(boxA[1], boxB[1]) - min(boxA[0], boxB[0])) * \
                     (max(boxA[3], boxB[3]) - min(boxA[2], boxB[2]))

        if merge_area == 0:
            return 0

        return float(intersection_area) / float(merge_area)

if __name__ == '__main__':
    eval = MattingEvaluation()
    omega = cv2.imread('D:/project/res_vis/alpha_fg_bg_comp/'
                     'archeology_00025.jpg_g0_34NEnnc_m_zewtrrkc.jpg_gt.png',
                     cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('D:/project/res_vis/alpha_fg_bg_comp/'
                     'archeology_00025.jpg_g0_34NEnnc_m_zewtrrkc.jpg_raw.png',
                     cv2.IMREAD_GRAYSCALE)
    print(eval.connectivity(img, omega))
