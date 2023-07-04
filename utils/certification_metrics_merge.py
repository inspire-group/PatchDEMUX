import numpy as np 
import torch 
import argparse
from pathlib import Path

from metrics import PerformanceMetrics

parser = argparse.ArgumentParser(description='Multi-Label PatchCleanser Certification Metrics Merge')

# Data information
parser.add_argument('--data-path', type=str)
parser.add_argument('--num-classes', default=80)

def file_print(file_path, msg):
    with open(file_path, "a") as f:
        print(msg, flush=True, file=f) 

def main():
    args = parser.parse_args()

    # Find number of subsets of data
    path = Path(args.data_path)
    subset_paths = [f for f in path.iterdir() if f.is_dir()]
    num_subsets = len(subset_paths)

    # Extract metrics data from each data subset
    metrics = PerformanceMetrics(args.num_classes)
    for folder in subset_paths:
        with np.load(str(folder / "certified_metrics.npz")) as file_metrics:
            metrics.updateMetrics(TP=file_metrics["TP"],TN=file_metrics["TN"],FN=file_metrics["FN"],FP=file_metrics["FP"])

    # Save the certified TP, TN, FN, FP
    np.savez(str(path / "certified_metrics_overall"), TP=metrics.TP, TN=metrics.TN, FN=metrics.FN, FP=metrics.FP)

    # Compute overall metrics
    precision_o, recall_o = metrics.overallPrecision(), metrics.overallRecall()
    precision_c, recall_c = metrics.averageClassPrecision(), metrics.averageClassRecall()
    all_class_precision = [metrics.classPrecision(i) for i in range(args.num_classes)]
    all_class_recall = [metrics.classRecall(i) for i in range(args.num_classes)]

    # Print to file
    logging_file = str(path / "performance_metrics_overall.txt")
    file_print(logging_file, f'P_O {precision_o:.2f} R_O {recall_o:.2f}\nP_C {precision_c:.2f} R_C {recall_c:.2f}\n')

    file_print(logging_file, f'CLASS        CLASS_PRECISION        CLASS_RECALL')
    file_print(logging_file, f'________________________________________________')
    for i in range(args.num_classes):
        file_print(logging_file, f'{i + 1:<13d}{metrics.classPrecision(i):<23.2f}{metrics.classRecall(i):.2f}')

if __name__ == '__main__':
    main()
