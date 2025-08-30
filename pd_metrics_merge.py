import numpy as np 
import argparse
from pathlib import Path

from utils.metrics import PerformanceMetrics
from utils.common import file_print

parser = argparse.ArgumentParser(description='Merge outputs from different data subsets when using non-cached PatchDEMUX scripts')

# Data information
parser.add_argument('--data-path', type=str, help='Path to directory containing data subset folders with metrics files')
parser.add_argument('--num-classes', default=80, type=int, help='Number of classes in the dataset')
parser.add_argument('--data-filename', default='certified_metrics', help='Base filename for metrics files')

def main():
    args = parser.parse_args()

    # Find number of subsets of data
    path = Path(args.data_path)
    subset_paths = [f for f in path.iterdir() if f.is_dir()]

    # Extract metrics data from each data subset - set args.num_classes to 1 for location-aware analysis
    num_classes = int(args.num_classes)
    metrics = PerformanceMetrics(num_classes)
    for folder in subset_paths:
        with np.load(str(folder / f"{args.data_filename}.npz")) as file_metrics:
            metrics.updateMetrics(TP=file_metrics["TP"],TN=file_metrics["TN"],FN=file_metrics["FN"],FP=file_metrics["FP"])

    # Save the certified TP, TN, FN, FP
    np.savez(str(path / f"{args.data_filename}_overall"), TP=metrics.TP, TN=metrics.TN, FN=metrics.FN, FP=metrics.FP)

    # Compute overall metrics
    precision_o, recall_o = metrics.overallPrecision(), metrics.overallRecall()
    precision_c, recall_c = metrics.averageClassPrecision(), metrics.averageClassRecall()

    # Print to file
    logging_file = str(path / "performance_metrics_overall.txt")
    file_print(logging_file, f'P_O {precision_o:.2f} R_O {recall_o:.2f} \n')
    file_print(logging_file, f'P_C {precision_c:.2f} R_C {recall_c:.2f}\n')

if __name__ == '__main__':
    main()
