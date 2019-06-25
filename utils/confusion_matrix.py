import torch
import math
import numpy as np
from sklearn.metrics import confusion_matrix


class ConfusionMatrix:
    def __init__(self, nclasses, classes, useUnlabeled=False):
        self.mat = np.zeros((nclasses, nclasses), dtype=np.float)
        self.valids = np.zeros((nclasses), dtype=np.float)
        self.IoU = np.zeros((nclasses), dtype=np.float)
        self.mIoU = 0

        self.nclasses = nclasses
        self.classes = classes
        self.list_classes = list(range(nclasses))
        self.useUnlabeled = useUnlabeled
        self.matStartIdx = 0

    def update_matrix(self, target, prediction):
        print('target ', target.shape)
        tar_x = target.unsqueeze(1)
        print('tar_x ', tar_x.shape)
        print('prediction ', prediction.shape, ' ', prediction)
        
        
        self.mat += confusion_matrix(tar_x, prediction)
     #   print('self.mat ', self.mat.shape, ' ', self.mat)

    def scores(self):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        total = 0   # Total true positives
        N = 0       # Total samples
        for i in range(self.matStartIdx, self.nclasses):
            N += sum(self.mat[:, i])
       #     print('N ', N)
            tp = self.mat[i][i]
       #     print('tp ', tp)
            fp = sum(self.mat[self.matStartIdx:, i]) - tp
       #     print('fp ', fp)
            fn = sum(self.mat[i,self.matStartIdx:]) - tp
       #     print('fn ', fn)
            if (tp+fp) == 0:
                self.valids[i] = 0
            else:
                self.valids[i] = tp/(tp + fp)

            if (tp+fp+fn) == 0:
                self.IoU[i] = 0
            else:
                self.IoU[i] = tp/(tp + fp + fn)

            total += tp

        self.mIoU = sum(self.IoU[self.matStartIdx:])/(self.nclasses - self.matStartIdx)
        self.accuracy = total/(sum(sum(self.mat[self.matStartIdx:, self.matStartIdx:])))

        return self.valids, self.accuracy, self.IoU, self.mIoU, self.mat

    def plot_confusion_matrix(self, filename):
        # Plot generated confusion matrix
        print(filename)


    def reset(self):
        self.mat = np.zeros((self.nclasses, self.nclasses), dtype=float)
        self.valids = np.zeros((self.nclasses), dtype=float)
        self.IoU = np.zeros((self.nclasses), dtype=float)
        self.mIoU = 0
