import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy
class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)

            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)  ##### pos 代表真值为正，neg代表真值为负
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)  ###class_pos 代表被分类为正值的总数

        False_alarm = (self.class_pos -self.tp_arr) / (self.pos_arr + self.neg_arr + 0.001)

        return tp_rates, fp_rates, recall, precision, False_alarm  #### tp_rates 代表Recall 值， fp_rate 代表 false positive rate

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])

        # print('reset_self.tp_arr:', self.tp_arr)

class PD_FA():
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.match_index = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.nIoU = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
    def update(self, preds, labels):
        labels = np.array((labels).cpu()).astype('int64')  # P

        b,c,h,w = labels.shape

        labelss = labels
        labelss = np.reshape(labelss, (b*c*1024, 1024))
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)
        for iBin in range(self.bins+1):
            score_thresh = (iBin ) / self.bins
            predit = np.array((preds > score_thresh).cpu()).astype('int64')
            predits = np.reshape(predit, (b*c*1024, 1024))


            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)


            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []
            self.IoU = 0
            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            self.sum_match=0
            self.match_index=[]
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))

                area_image = np.array(coord_image[m].area)

                for i in range(len(coord_label)):
                    centroid_label = np.array(list(coord_label[i].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    if distance < 0.5*coord_label[i].equivalent_diameter:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)
                        self.match_index.append(i)
                        self.sum_match=self.sum_match+1

                        intersection = np.sum(np.array(image==m+1)*np.array(label==i+1))
                        label_sum = np.sum(np.array(coord_label[i].area))
                        pred_sum = np.sum(area_image)
                        self.IoU += intersection/(label_sum+pred_sum-intersection)
                        # del coord_image[m]
                        break

            self.match_index= list(set(self.match_index))

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.match_index)
            self.nIoU[iBin]+=self.IoU


    def get(self,img_num):

        Final_FA =  self.FA / ((1024 * 1024) * img_num)
        Final_PD =  self.PD /self.target
        Final_nIoU = self.nIoU / self.target

        return Final_FA,Final_PD,Final_nIoU


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])
        self.nIoU= np.zeros([self.bins+1])
        self.target = np.zeros(self.bins + 1)

class mIoU():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        # print('self.total_correct:',self.total_correct)

    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0




def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category 0 is ignored class, typically for background / boundary

    predict = (output > score_thresh).float()  # P
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1) # T
    elif len(target.shape) == 4:
        target = target.float()                  # T
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())  # TP

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()  # FP
    tn = ((1 - predict) * ((predict == target).float())).sum()  # TN
    fn = (((predict != target).float()) * (1 - predict)).sum()   # FN
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos          ## pos 代表真值为正，neg代表真值为负, class_pos代表被分类为正值

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1) # T
    elif len(target.shape) == 4:
        target = target.float() # T
    else:
        raise ValueError("Unknown target dimension")
    # print("output.shape: ", output.shape)
    # print("target.shape: ", target.shape)
    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float() # P
    pixel_labeled = (target > 0).float().sum() # T
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()# TP

    # predict = predict .reshape(8, 512, 512)
    # plt.imshow(predict[0])
    # plt.show()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category 0 is ignored class, typically for background / boundary
    mini = 1
    maxi = 1  # nclass
    nbins = 1  # nclass
    predict = (output > 0).float()  # P
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1) # T
    elif len(target.shape) == 4:
        target = target.float()  # T
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())  # TP

    # areas of intersection and union
    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter
    # print('area_inter:', area_inter)
    # print('area_pred:', area_pred)
    # print('area_lab:', area_lab)
    # print('area_union:', area_union)
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union

# ######################################################
# ###1.测试ROC好使不
# ROC = ROCMetric(1, 10)
# # pred=np.random.randn(4,1,480,480)
# # labels=np.random.randint(0,2,size=(4,1,480,480))
# # np.save('pred.npy'  ,pred)
# # np.save('labels.npy',labels)
# pred=np.load('pred.npy')
# labels=np.load('labels.npy')
# pred=torch.from_numpy(pred)
# labels=torch.from_numpy(labels)
#
# ROC.update(pred, labels)
# ture_positive_rate, false_positive_rate, recall, precision, False_alarm =ROC.get()
# print(ture_positive_rate)
# ######################################################

# # ######################################################
# # # ###2.测试mIoU好使不
# mIoU =mIoU (1)
# # pred=np.random.randn(4,1,480,480)
# # labels=np.random.randint(0,2,size=(4,1,480,480))
# # np.save('pred.npy'  ,pred)
# # np.save('labels.npy',labels)
# pred=np.load('pred.npy')
# labels=np.load('labels.npy')
# pred=torch.from_numpy(pred)
# labels=torch.from_numpy(labels)
#
# mIoU.update(pred, labels)
# pixAcc, mIoU=mIoU.get()
# print(pixAcc, mIoU)

