# Import necessary libraries
import os
import pickle

import matplotlib
import pandas as pd

# Use 'Agg' backend for matplotlib to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
import cv2
import numpy as np
import torch
from skimage import transform as trans
from backbones import get_model
from sklearn.metrics import roc_curve, auc

from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path

import sys
import warnings

# Insert the parent directory to the system path to access other modules
sys.path.insert(0, "../")
# Ignore all warnings
warnings.filterwarnings("ignore")

# Define an argument parser to handle command-line arguments
parser = argparse.ArgumentParser(description='do ijb test')
# general
parser.add_argument('--model-prefix', default='', help='path to load model.')
parser.add_argument('--image-path', default='', type=str, help='path to images')
parser.add_argument('--result-dir', default='.', type=str, help='directory to save results')
parser.add_argument('--batch-size', default=128, type=int, help='batch size for processing')
parser.add_argument('--network', default='iresnet50', type=str, help='network architecture')
parser.add_argument('--job', default='insightface', type=str, help='job name')
parser.add_argument('--target', default='IJBC', type=str, help='target dataset, set to IJBC or IJBB')
args = parser.parse_args()

# Extract arguments
target = args.target
model_path = args.model_prefix
image_path = args.image_path
result_dir = args.result_dir
gpu_id = None
use_norm_score = True  # if True, TestMode(N1)
use_detector_score = True  # if True, TestMode(D1)
use_flip_test = True  # if True, TestMode(F1)
job = args.job
batch_size = args.batch_size


class Embedding(object):
    """
    A class to handle the embedding extraction from images using a pre-trained model.
    """
    def __init__(self, prefix, data_shape, batch_size=1):
        """
        Initialize the Embedding class.

        Args:
            prefix (str): Path to the model weights.
            data_shape (tuple): Shape of the input data.
            batch_size (int, optional): Batch size for processing. Defaults to 1.
        """
        image_size = (112, 112)
        self.image_size = image_size
        weight = torch.load(prefix, map_location='cuda:0')
        resnet = get_model(args.network, dropout=0, fp16=False).cuda()
        
        print(resnet)
        
        # Load the model weights
        resnet.load_state_dict(weight["state_dict_backbone"])
        model = torch.nn.DataParallel(resnet)
        self.model = model
        self.model.eval()
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape

    def get(self, rimg, landmark):
        """
        Preprocess the image and extract the embedding.

        Args:
            rimg (numpy.ndarray): Raw image.
            landmark (numpy.ndarray): Landmark points for alignment.

        Returns:
            numpy.ndarray: Preprocessed image batch.
        """
        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg,
                             M, (self.image_size[1], self.image_size[0]),
                             borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data):
        """
        Forward pass through the model to get embeddings.

        Args:
            batch_data (numpy.ndarray): Batch of images.

        Returns:
            numpy.ndarray: Embeddings.
        """
        imgs = torch.Tensor(batch_data).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        
        print(imgs.shape)
        
        _, _, feat = self.model(imgs)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()


def divideIntoNstrand(listTemp, n):
    """
    Divide a list into n strands.

    Args:
        listTemp (list): List to be divided.
        n (int): Number of strands.

    Returns:
        list: List of strands.
    """
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList


def read_template_media_list(path):
    """
    Read template media list from a file.

    Args:
        path (str): Path to the file.

    Returns:
        tuple: Templates and medias.
    """
    # ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def read_template_pair_list(path):
    """
    Read template pair list from a file.

    Args:
        path (str): Path to the file.

    Returns:
        tuple: Template pairs and labels.
    """
    # pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


def read_image_feature(path):
    """
    Read image features from a file.

    Args:
        path (str): Path to the file.

    Returns:
        numpy.ndarray: Image features.
    """
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


def get_image_feature(img_path, files_list, model_path, epoch, gpu_id):
    """
    Extract features from images.

    Args:
        img_path (str): Path to images.
        files_list (list): List of image files.
        model_path (str): Path to the model.
        epoch (int): Epoch number.
        gpu_id (int): GPU ID.

    Returns:
        tuple: Image features and faceness scores.
    """
    batch_size = args.batch_size
    data_shape = (3, 112, 112)

    files = files_list
    print('files:', len(files))
    rare_size = len(files) % batch_size
    faceness_scores = []
    batch = 0
    img_feats = np.empty((len(files), 1024), dtype=np.float32)

    batch_data = np.empty((2 * batch_size, 3, 112, 112))
    embedding = Embedding(model_path, data_shape, batch_size)
    for img_index, each_line in enumerate(files[:len(files) - rare_size]):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk)

        batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]
        batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]
        if (img_index + 1) % batch_size == 0:
            print('batch', batch)
            img_feats[batch * batch_size:batch * batch_size +
                                         batch_size][:] = embedding.forward_db(batch_data)
            batch += 1
        faceness_scores.append(name_lmk_score[-1])

    batch_data = np.empty((2 * rare_size, 3, 112, 112))
    embedding = Embedding(model_path, data_shape, rare_size)
    for img_index, each_line in enumerate(files[len(files) - rare_size:]):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk)
        batch_data[2 * img_index][:] = input_blob[0]
        batch_data[2 * img_index + 1][:] = input_blob[1]
        if (img_index + 1) % rare_size == 0:
            print('batch', batch)
            img_feats[len(files) -
                      rare_size:][:] = embedding.forward_db(batch_data)
            batch += 1
        faceness_scores.append(name_lmk_score[-1])
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    return img_feats, faceness_scores


def image2template_feature(img_feats=None, templates=None, medias=None):
    """
    Convert image features to template features.

    Args:
        img_feats (numpy.ndarray): Image features.
        templates (numpy.ndarray): Templates.
        medias (numpy.ndarray): Medias.

    Returns:
        tuple: Template features and unique templates.
    """
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    """
    Perform verification between template pairs.

    Args:
        template_norm_feats (numpy.ndarray): Normalized template features.
        unique_templates (numpy.ndarray): Unique templates.
        p1 (numpy.ndarray): Template pairs.
        p2 (numpy.ndarray): Template pairs.

    Returns:
        numpy.ndarray: Verification scores.
    """
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def verification2(template_norm_feats=None,
                  unique_templates=None,
                  p1=None,
                  p2=None):
    """
    Perform verification between template pairs (alternative implementation).

    Args:
        template_norm_feats (numpy.ndarray): Normalized template features.
        unique_templates (numpy.ndarray): Unique templates.
        p1 (numpy.ndarray): Template pairs.
        p2 (numpy.ndarray): Template pairs.

    Returns:
        numpy.ndarray: Verification scores.
    """
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    """
    Read scores from a file.

    Args:
        path (str): Path to the file.

    Returns:
        numpy.ndarray: Scores.
    """
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# Ensure the target is either 'IJBC' or 'IJBB'
assert target == 'IJBC' or target == 'IJBB'

# Measure time taken to read template media list
start = timeit.default_timer()
templates, medias = read_template_media_list(
    os.path.join('%s/meta' % image_path,
                 '%s_face_tid_mid.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# Measure time taken to read template pair list
start = timeit.default_timer()
p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % image_path,
                 '%s_template_pair_label.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# Measure time taken to extract image features
start = timeit.default_timer()
img_path = '%s/loose_crop' % image_path
img_list_path = '%s/meta/%s_name_5pts_score.txt' % (image_path, target.lower())
img_list = open(img_list_path)
files = img_list.readlines()
files_list = files

# Extract image features
img_feats, faceness_scores = get_image_feature(img_path, files_list,
                                               model_path, 0, gpu_id)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                          img_feats.shape[1]))

# Measure time taken to process image features
start = timeit.default_timer()

if use_flip_test:
    # Combine features from original and flipped images
    img_input_feats = img_feats[:, 0:img_feats.shape[1] //
                                     2] + img_feats[:, img_feats.shape[1] // 2:]
else:
    # Use only the original image features
    img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]

if use_norm_score:
    # Use normalized scores
    img_input_feats = img_input_feats
else:
    # Normalize features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(
        np.sum(img_input_feats ** 2, -1, keepdims=True))

if use_detector_score:
    # Use detector scores
    print(img_input_feats.shape, faceness_scores.shape)
    img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
else:
    # Do not use detector scores
    img_input_feats = img_input_feats

# Convert image features to template features
template_norm_feats, unique_templates = image2template_feature(
    img_input_feats, templates, medias)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# Measure time taken to perform verification
start = timeit.default_timer()
score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# Save the verification scores
save_path = os.path.join(result_dir, args.job)
if not os.path.exists(save_path):
    os.makedirs(save_path)

score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
np.save(score_save_file, score)

# Plot ROC curves and generate TPR@FPR table
files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))
colours = dict(
    zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
fig = plt.figure()
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' %
                    (method.split('-')[-1], roc_auc * 100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
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
fig.savefig(os.path.join(save_path, '%s.pdf' % target.lower()))
print(tpr_fpr_table)