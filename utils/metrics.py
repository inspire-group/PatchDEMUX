import numpy as np

class PerformanceMetrics:
    def __init__(self, num_classes):
        if (num_classes < 1):
            raise ValueError('num_classes must be at least 1')

        self.num_classes = num_classes
        self.TP = np.zeros((num_classes,))
        self.TN = np.zeros((num_classes,))
        self.FN = np.zeros((num_classes,))
        self.FP = np.zeros((num_classes,))        

    # Update the TP, TN, FN, and FP values
    def updateMetrics(self, TP, TN, FN, FP):
        # If num_classes > 1: each metric assumed to be array of shape (num_values, num_classes) or list of length num_classes
        # If num_classes = 1: each metric assumed to be array of shape (num_values, 1) or a scalar
        def preprocessInput(metric):
            np_metric = np.array(metric)
            return np_metric.flatten() if np_metric.ndim <= 1 else np_metric.sum(axis=0).flatten()
        
        self.TP += preprocessInput(TP)
        self.TN += preprocessInput(TN)
        self.FN += preprocessInput(FN)
        self.FP += preprocessInput(FP)

    # Compute overall precision
    def overallPrecision(self):
        total_TP = self.TP.sum()
        total_FP = self.FP.sum()
        return total_TP / (total_TP + total_FP) * 100 if total_TP > 0 else 0.0

    # Compute overall recall
    def overallRecall(self):
        total_TP = self.TP.sum()
        total_FN = self.FN.sum()
        return total_TP / (total_TP + total_FN) * 100 if total_TP > 0 else 0.0

    # Compute overall false positive rate
    def overallFPR(self):
        total_FP = self.FP.sum()
        total_TN = self.TN.sum()
        return total_FP / (total_FP + total_TN) * 100 if total_FP > 0 else 0.0

    # Compute precision of a single class
    def classPrecision(self, index):
        if (not isinstance(index, int)):
            raise ValueError('Index must be an integer')

        class_TP = self.TP[index]
        class_FP = self.FP[index]
        return class_TP / (class_TP + class_FP) * 100 if class_TP > 0 else 0.0

    # Compute recall of a single class
    def classRecall(self, index):
        if (not isinstance(index, int)):
            raise ValueError('Index must be an integer')
            
        class_TP = self.TP[index]
        class_FN = self.FN[index]
        return class_TP / (class_TP + class_FN) * 100 if class_TP > 0 else 0.0
    
    # Compute false positive rate of a single class
    def classFPR(self, index):
        if (not isinstance(index, int)):
            raise ValueError('Index must be an integer')
            
        class_FP = self.FP[index]
        class_TN = self.TN[index]
        return class_FP / (class_FP + class_TN) * 100 if class_FP > 0 else 0.0

    # Compute average precision over all classes
    def averageClassPrecision(self):
        precision = np.zeros((self.num_classes, ))

        # Only account for classes with a non-zero TP within the calculation of the average
        nonzeroTP = self.TP > 0
        if (not nonzeroTP.any()):
            return 0.0
        precision[nonzeroTP] = self.TP[nonzeroTP] / (self.TP[nonzeroTP] + self.FP[nonzeroTP]) * 100.0
        return np.mean(precision)

    # Compute average recall over all classes
    def averageClassRecall(self):
        recall = np.zeros((self.num_classes, ))

        # Only account for classes with a non-zero TP within the calculation of the average
        nonzeroTP = self.TP > 0
        if (not nonzeroTP.any()):
            return 0.0
        recall[nonzeroTP] = self.TP[nonzeroTP] / (self.TP[nonzeroTP] + self.FN[nonzeroTP]) * 100.0
        return np.mean(recall)