import os  # For interacting with the operating system
import sys  # For accessing command-line arguments

import matplotlib.pyplot as plt  # For plotting graphs
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap  # For sampling colors from a colormap
from prettytable import PrettyTable  # For creating and printing pretty tables
from sklearn.metrics import roc_curve, auc  # For computing ROC curve and AUC

# Read the list of score files from the command-line argument
with open(sys.argv[1], "r") as f:
    files = f.readlines()

# Strip whitespace from each line and store in a list
files = [x.strip() for x in files]
image_path = "/train_tmp/IJB_release/IJBC"  # Path to the image dataset


def read_template_pair_list(path):
    """
    Reads template pair list from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        t1 (numpy.ndarray): Template 1 IDs.
        t2 (numpy.ndarray): Template 2 IDs.
        label (numpy.ndarray): Labels for the pairs.
    """
    # Read the CSV file and convert to numpy array
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int)  # Template 1 IDs
    t2 = pairs[:, 1].astype(np.int)  # Template 2 IDs
    label = pairs[:, 2].astype(np.int)  # Labels
    return t1, t2, label


# Read the template pair list for IJBC
p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % image_path,
                 '%s_template_pair_label.txt' % 'ijbc'))

# Initialize lists to store methods and scores
methods = []
scores = []
for file in files:
    methods.append(file)  # Store the method name
    scores.append(np.load(file))  # Load the score file

# Convert methods to a numpy array and create a dictionary for scores
methods = np.array(methods)
scores = dict(zip(methods, scores))

# Sample colors for each method from a colormap
colours = dict(
    zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))

# Define x-axis labels for the ROC plot
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

# Initialize a PrettyTable for displaying TPR at specific FPR values
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])

# Create a figure for plotting
fig = plt.figure()

# Loop through each method to compute and plot ROC curves
for method in methods:
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    
    # Flip arrays to ensure correct order
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # Select largest TPR at the same FPR
    
    # Plot the ROC curve
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' % (method.split('-')[-1], roc_auc * 100)))
    
    # Prepare a row for the TPR-FPR table
    tpr_fpr_row = []
    tpr_fpr_row.append(method)
    for fpr_iter in np.arange(len(x_labels)):
        # Find the index of the closest FPR value
        _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    
    # Add the row to the table
    tpr_fpr_table.add_row(tpr_fpr_row)

# Set plot limits and labels
plt.xlim([10 ** -6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")

# Print the TPR-FPR table
print(tpr_fpr_table)