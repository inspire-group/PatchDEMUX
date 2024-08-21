# TODO:
# - CHECK, BUT GIVEN USE OF EXCEL FOR PLOTTING THIS SCRIPT MIGHT NOT BE NEEDED AND CAN BE DELETED

import numpy as np 
import matplotlib.pyplot as plt
import torch 
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Multi-Label PatchCleanser ROC Plot')

# Data information
parser.add_argument('--data-path', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('--class-id', type=int, default=0)

# Pick which metrics plot to create
parser.add_argument('--metric_plot', type=str, choices=["roc", "prec_recall"], default="roc")

def file_print(file_path, msg):
    with open(file_path, "a") as f:
        print(msg, flush=True, file=f) 

def save_metrics_plot(first_metric, second_metric, first_metric_name, second_metric_name, title, save_path):
    # Create metrics plot
    plt.figure()
    plt.scatter(first_metric, second_metric, label=f"{first_metric_name}_and_{second_metric_name}_plot")    # Metrics plot
    #plt.plot(np.arange(100), np.arange(100), "y--", label="Random classifier")    # 1:1 line associated with random classifier
    
    # Plot formatting
    plt.title(title)
    plt.xlabel(f"{first_metric_name}")
    plt.ylabel(f"{second_metric_name}")
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.legend()
    
    # Save the plot
    plt.savefig(save_path) 

def main():
    args = parser.parse_args()

    # Find number of data subsets with different thresholding values
    path = Path(args.data_path)
    subset_paths = [f for f in path.iterdir() if f.is_dir()]
    num_subsets = len(subset_paths)

    # Extract data from each subset
    overall_first_metric_array = np.zeros(num_subsets)
    overall_second_metric_array = np.zeros(num_subsets)

    class_id_first_metric_array = np.zeros(num_subsets)
    class_id_second_metric_array = np.zeros(num_subsets)

    # Define the metrics
    if args.metric_plot == "roc":
        first_metric, second_metric = "fpr_o", "recall_o"
        first_metric_class, second_metric_class = "all_class_fpr", "all_class_recall"

    elif args.metric_plot == "prec_recall":
        first_metric, second_metric = "recall_o", "precision_o"
        first_metric_class, second_metric_class = "all_class_recall", "all_class_precision"

    # Load the metrics data
    for index, folder in enumerate(subset_paths):
        with np.load(str(folder / f"{args.metric_plot}_data.npz")) as metrics_data:
            overall_first_metric_array[index] = metrics_data[first_metric]
            overall_second_metric_array[index] = metrics_data[second_metric]

            class_id_first_metric_array[index] = metrics_data[first_metric_class][args.class_id]
            class_id_second_metric_array[index] = metrics_data[second_metric_class][args.class_id]

    title_header = f"{args.metric_plot} plot"

    breakpoint()

    # Create metrics plot
    save_metrics_plot(overall_first_metric_array, overall_second_metric_array, first_metric, second_metric, f"{title_header} plot for overall classification", str(path / f"{args.metric_plot}_plot.png"))

    # Create metrics plot for class id
    save_metrics_plot(class_id_first_metric_array, class_id_second_metric_array, first_metric_class, second_metric_class, f"{title_header} plot for class {args.class_id} of {args.num_classes}", str(path / f"{args.metric_plot}_plot_class_id_{args.class_id}.png"))

if __name__ == '__main__':
    main()
