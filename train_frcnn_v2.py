from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os
import re
import shutil

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers

from tensorflow.python.keras.utils import generic_utils

sys.setrecursionlimit(40000)

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

# if Logs path directory exists, it will delete the directory
if os.path.exists('logs'):
    shutil.rmtree('logs')

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-v", "--valid_path", dest="valid_path", help="Path to validation data.")
parser.add_option("-o", "--parser", dest="parser",
                  help="Parser to use. One of simple or pascal_voc", default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                  help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network",
                  help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--hf", dest="horizontal_flips",
                  help="Augment with horizontal flips in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips",
                  help="Augment with vertical flips in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90",
                  help="Augment with 90 degree rotations in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--num_epochs", type="int",
                  dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--config_filename", dest="config_filename",
                  help="Location to store all the metadata related to "
                       "the training (to be used when testing).",
                  default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path",
                  help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path",
                  help="Input path for weights. If not specified, will try to"
                       " load default weights provided by keras.")

(options, args) = parser.parse_args()

if not options.train_path:  # if filename is not given
    parser.error('Error: path to training data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.model_path = options.output_weight_path
model_path_regex = re.match("^(.+)(\.hdf5)$", C.model_path)
if model_path_regex.group(2) != '.hdf5':
    print('Output weights must have .hdf5 filetype')
    exit(1)
C.num_rois = int(options.num_rois)

if options.network == 'vgg':
    C.network = 'vgg'
    from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
    from keras_frcnn import resnet as nn

    C.network = 'resnet50'
else:
    print('Not a valid model')
    raise ValueError

# check if weight path was passed via command line
if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    # set the path to weights based on backend and model
    C.base_net_weights = nn.get_weight_path()

train_imgs, classes_count, class_mapping = get_data(options.train_path)
val_imgs, _, _ = get_data(options.valid_path)

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print(f'Num classes (including bg) = {len(classes_count)}')

config_output_filename = options.config_filename
with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print(f'Config has been written to {config_output_filename}, '
          f'and can be loaded when testing to ensure correct results')

num_imgs = len(train_imgs)
num_valid_imgs = len(val_imgs)

print(f'Num train samples {len(train_imgs)}')
print(f'Num val samples {len(val_imgs)}')

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C,
                                               nn.get_img_output_length,
                                               K.image_data_format(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,
                                             K.image_data_format(), mode='val')

if K.image_data_format() == 'channels_first':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)
classifier = nn.classifier(shared_layers, roi_input, C.num_rois,
                           nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier,
# used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

# Defining optimizers for all models
optimizer_rpn = Adam(learning_rate=1e-5)
optimizer_classifier = Adam(learning_rate=1e-5)
optimizer_all = SGD(learning_rate=0.01)

# Accuracy metrics for Fast RCNN model
train_classifier_metric = tf.keras.metrics.CategoricalAccuracy()
val_classifier_metric = tf.keras.metrics.CategoricalAccuracy()

# Loss function of RPN model and Fast RCNN model
rpn_class_loss_fn = losses.RpnClassificationLoss(num_anchors)
rpn_reg_loss_fn = losses.RpnRegressionLoss(num_anchors)
fast_rcnn_class_loss_fn = losses.FastrcnnClassLoss()
fast_rcnn_reg_loss_fn = losses.FastrcnnRegLoss(len(classes_count) - 1)

# tensorboard writer, automatically creates directory and writes logs
train_writer = tf.summary.create_file_writer('logs/train/')
valid_writer = tf.summary.create_file_writer('logs/valid/')


@tf.function
def rpn_train_step(step, x_batch_train, y_batch_train):
    with tf.GradientTape() as rpn_tape:
        y_rpn_cls_true, y_rpn_regr_true = y_batch_train
        y_rpn_cls_pred, y_rpn_regr_pred = model_rpn(x_batch_train, training=True)

        rpn_class_loss = rpn_class_loss_fn(y_rpn_cls_true, y_rpn_cls_pred)
        rpn_reg_loss = rpn_reg_loss_fn(y_rpn_regr_true, y_rpn_regr_pred)

    rpn_grads = rpn_tape.gradient([rpn_class_loss, rpn_reg_loss],
                                  model_rpn.trainable_weights)
    optimizer_rpn.apply_gradients(zip(rpn_grads, model_rpn.trainable_weights))

    # write training loss and accuracy to the tensorboard
    with train_writer.as_default():
        tf.summary.scalar('rpn_class_loss', rpn_class_loss, step=step)
        tf.summary.scalar('rpn_reg_loss', rpn_reg_loss, step=step)

    return y_rpn_cls_pred, y_rpn_regr_pred, rpn_class_loss, rpn_reg_loss


@tf.function
def frcnn_train_step(step, x_batch_train, X2, Y1, Y2):
    with tf.GradientTape() as frcnn_tape:
        rcnn_class_pred, rcnn_reg_pred = model_classifier([x_batch_train, X2],
                                                          training=True)

        fast_rcnn_class_loss = fast_rcnn_class_loss_fn(Y1, rcnn_class_pred)
        fast_rcnn_reg_loss = fast_rcnn_reg_loss_fn(Y2, rcnn_reg_pred)

    frcnn_grads = frcnn_tape.gradient([fast_rcnn_class_loss, fast_rcnn_reg_loss],
                                      model_classifier.trainable_weights)
    optimizer_classifier.apply_gradients(zip(frcnn_grads, model_classifier.trainable_weights))
    train_classifier_metric.update_state(Y1, rcnn_class_pred)
    fast_rcnn_class_acc = train_classifier_metric.result()

    # write training loss and accuracy to the tensorboard
    with train_writer.as_default():
        tf.summary.scalar('fast_rcnn_class_loss', fast_rcnn_class_loss, step=step)
        tf.summary.scalar('fast_rcnn_reg_loss', fast_rcnn_reg_loss, step=step)
        tf.summary.scalar('fast_rcnn_class_acc', fast_rcnn_class_acc, step=step)

    return fast_rcnn_class_loss, fast_rcnn_reg_loss, fast_rcnn_class_acc


@tf.function
def rpn_valid_step(step, x_batch_train, y_batch_train):
    with tf.GradientTape() as rpn_tape:
        y_rpn_cls_true, y_rpn_regr_true = y_batch_train
        y_rpn_cls_pred, y_rpn_regr_pred = model_rpn(x_batch_train, training=False)

        rpn_class_loss = rpn_class_loss_fn(y_rpn_cls_true, y_rpn_cls_pred)
        rpn_reg_loss = rpn_reg_loss_fn(y_rpn_regr_true, y_rpn_regr_pred)

    # write training loss and accuracy to the tensorboard
    with valid_writer.as_default():
        tf.summary.scalar('rpn_class_loss', rpn_class_loss, step=step)
        tf.summary.scalar('rpn_reg_loss', rpn_reg_loss, step=step)

    return y_rpn_cls_pred, y_rpn_regr_pred, rpn_class_loss, rpn_reg_loss


@tf.function
def frcnn_valid_step(step, x_batch_train, X2, Y1, Y2):
    with tf.GradientTape() as frcnn_tape:
        rcnn_class_pred, rcnn_reg_pred = model_classifier([x_batch_train, X2],
                                                          training=False)
        fast_rcnn_class_loss = fast_rcnn_class_loss_fn(Y1, rcnn_class_pred)
        fast_rcnn_reg_loss = fast_rcnn_reg_loss_fn(Y2, rcnn_reg_pred)

    val_classifier_metric.update_state(Y1, rcnn_class_pred)
    fast_rcnn_class_acc = val_classifier_metric.result()

    # write training loss and accuracy to the tensorboard
    with valid_writer.as_default():
        tf.summary.scalar('fast_rcnn_class_loss', fast_rcnn_class_loss, step=step)
        tf.summary.scalar('fast_rcnn_reg_loss', fast_rcnn_reg_loss, step=step)
        tf.summary.scalar('fast_rcnn_class_acc', fast_rcnn_class_acc, step=step)

    return fast_rcnn_class_loss, fast_rcnn_reg_loss, fast_rcnn_class_acc


def get_selected_samples(Y1, rpn_accuracy_rpn_monitor, rpn_accuracy_for_epoch):
    neg_samples = np.where(Y1[0, :, -1] == 1)
    pos_samples = np.where(Y1[0, :, -1] == 0)

    if len(neg_samples) > 0:
        neg_samples = neg_samples[0]
    else:
        neg_samples = []

    if len(pos_samples) > 0:
        pos_samples = pos_samples[0]
    else:
        pos_samples = []

    rpn_accuracy_rpn_monitor.append(len(pos_samples))
    rpn_accuracy_for_epoch.append((len(pos_samples)))

    if C.num_rois > 1:
        if len(pos_samples) < C.num_rois // 2:
            selected_pos_samples = pos_samples.tolist()
        else:
            selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2,
                                                    replace=False).tolist()
        try:
            selected_neg_samples = np.random.choice(neg_samples,
                                                    C.num_rois - len(selected_pos_samples),
                                                    replace=False).tolist()
        except:
            selected_neg_samples = np.random.choice(neg_samples,
                                                    C.num_rois - len(selected_pos_samples),
                                                    replace=True).tolist()

        sel_samples = selected_pos_samples + selected_neg_samples
    else:
        # in the extreme case where num_rois = 1, we pick a random pos or neg sample
        selected_pos_samples = pos_samples.tolist()
        selected_neg_samples = neg_samples.tolist()
        if np.random.randint(0, 2):
            sel_samples = random.choice(neg_samples)
        else:
            sel_samples = random.choice(pos_samples)

    return sel_samples


n_epochs = options.num_epochs
BATCH_SIZE = 1
n_steps = num_imgs // BATCH_SIZE
n_valid_steps = num_valid_imgs // BATCH_SIZE

losses = np.zeros((n_steps, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []

valid_losses = np.zeros((n_valid_steps, 5))
rpn_accuracy_rpn_monitor_valid = []
rpn_accuracy_for_epoch_valid = []

best_loss = np.Inf
start_time = time.time()

class_mapping_inv = {v: k for k, v in class_mapping.items()}

global_step = tf.convert_to_tensor(0, tf.int64)
one_step = tf.convert_to_tensor(1, tf.int64)

print("Training started for %d epochs" % n_epochs)
for epoch in range(n_epochs):
    print("\nStart of epoch %d" % (epoch + 1,))
    progbar = generic_utils.Progbar(n_steps)

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train, img_data) in enumerate(data_gen_train):
        # print(step, img_data['filepath'])

        y_rpn_cls_true, y_rpn_regr_true = y_batch_train
        step = tf.cast(step, dtype=tf.int64)
        global_step = tf.add(global_step, one_step)
        y_rpn_cls_pred, y_rpn_regr_pred, rpn_class_loss, rpn_reg_loss = rpn_train_step(
            global_step, x_batch_train, y_batch_train)

        R = roi_helpers.rpn_to_roi(y_rpn_cls_pred, y_rpn_regr_pred, C, K.image_data_format(),
                                   use_regr=True, overlap_thresh=0.7, max_boxes=300)
        # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
        X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

        if X2 is None:
            rpn_accuracy_rpn_monitor.append(0)
            rpn_accuracy_for_epoch.append(0)
            continue

        sel_samples = get_selected_samples(Y1, rpn_accuracy_rpn_monitor, rpn_accuracy_for_epoch)

        x2_tensor = tf.convert_to_tensor(X2[:, sel_samples, :], tf.float32)
        y1_tensor = tf.convert_to_tensor(Y1[:, sel_samples, :], tf.float32)
        y2_tensor = tf.convert_to_tensor(Y2[:, sel_samples, :], tf.float32)

        fast_rcnn_class_loss, fast_rcnn_reg_loss, fast_rcnn_class_acc = frcnn_train_step(
            global_step, x_batch_train, x2_tensor, y1_tensor, y2_tensor)

        losses[step, 0] = rpn_class_loss
        losses[step, 1] = rpn_reg_loss

        losses[step, 2] = fast_rcnn_class_loss
        losses[step, 3] = fast_rcnn_reg_loss
        losses[step, 4] = fast_rcnn_class_acc

        progbar.update(step + 1,
                       [('rpn_cls', rpn_class_loss),
                        ('rpn_regr', rpn_reg_loss),
                        ('detector_cls', fast_rcnn_class_loss),
                        ('detector_regr', fast_rcnn_reg_loss)])

        if step == n_steps - 1 and C.verbose:
            mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)
                                            ) / len(rpn_accuracy_rpn_monitor)
            rpn_accuracy_rpn_monitor = []
            print(f'\nAverage number of overlapping bounding boxes '
                  f'from RPN = {mean_overlapping_bboxes} for {step} previous iterations')
            if mean_overlapping_bboxes == 0:
                print('RPN is not producing bounding boxes that overlap the ground truth boxes.'
                      ' Check RPN settings or keep training.')

            loss_rpn_cls = np.mean(losses[:, 0])
            loss_rpn_regr = np.mean(losses[:, 1])
            loss_class_cls = np.mean(losses[:, 2])
            loss_class_regr = np.mean(losses[:, 3])
            class_acc = np.mean(losses[:, 4])

            mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(
                rpn_accuracy_for_epoch)
            rpn_accuracy_for_epoch = []

            if C.verbose:
                print(
                    f'\nMean number of bounding boxes from RPN overlapping '
                    f'ground truth boxes: {mean_overlapping_bboxes}')
                print(f'Classifier accuracy for bounding boxes from RPN: {class_acc}')
                print(f'Loss RPN classifier: {loss_rpn_cls}')
                print(f'Loss RPN regression: {loss_rpn_regr}')
                print(f'Loss Detector classifier: {loss_class_cls}')
                print(f'Loss Detector regression: {loss_class_regr}')
                print(f'Elapsed time: {time.time() - start_time}')

            curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
            print("Total Loss: %.4f" % curr_loss)
            start_time = time.time()

            if curr_loss < best_loss:
                if C.verbose:
                    print(
                        f'Total loss decreased from {best_loss} to {curr_loss}, saving weights')
                best_loss = curr_loss

            model_all.save_weights(model_path_regex.group(1) + "_" + '{:04d}'.format(
                epoch) + model_path_regex.group(2))
            break

        # # Log every 10 steps.
        # if step % 10 == 0:
        #     print("Step %d, RPN Cls Loss: %.4f RPN reg Loss: %.4f "
        #           "FRCNN Cls Loss: %.4f FRCNN reg Loss: %.4f" % (
        #            step, float(rpn_class_loss), float(rpn_reg_loss), float(fast_rcnn_class_loss),
        #               float(fast_rcnn_reg_loss)))

    # Reset training metrics at the end of each epoch
    train_classifier_metric.reset_states()

    progbar = generic_utils.Progbar(n_valid_steps)
    # Iterate over the batches of the dataset.
    for step, (x_batch_val, y_batch_val, img_data) in enumerate(data_gen_val):

        y_rpn_cls_true, y_rpn_regr_true = y_batch_val
        y_rpn_cls_pred, y_rpn_regr_pred, rpn_class_loss, rpn_reg_loss = rpn_valid_step(
            global_step, x_batch_val, y_batch_val)

        R = roi_helpers.rpn_to_roi(y_rpn_cls_pred, y_rpn_regr_pred, C, K.image_data_format(),
                                   use_regr=True, overlap_thresh=0.7, max_boxes=300)

        # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
        X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

        if X2 is None:
            rpn_accuracy_rpn_monitor_valid.append(0)
            rpn_accuracy_for_epoch_valid.append(0)
            continue

        sel_samples = get_selected_samples(Y1, rpn_accuracy_rpn_monitor_valid,
                                           rpn_accuracy_for_epoch_valid)

        x2_tensor = tf.convert_to_tensor(X2[:, sel_samples, :], tf.float32)
        y1_tensor = tf.convert_to_tensor(Y1[:, sel_samples, :], tf.float32)
        y2_tensor = tf.convert_to_tensor(Y2[:, sel_samples, :], tf.float32)

        fast_rcnn_class_loss, fast_rcnn_reg_loss, fast_rcnn_class_acc = frcnn_valid_step(
            global_step, x_batch_val, x2_tensor, y1_tensor, y2_tensor)

        valid_losses[step, 0] = rpn_class_loss
        valid_losses[step, 1] = rpn_reg_loss

        valid_losses[step, 2] = fast_rcnn_class_loss
        valid_losses[step, 3] = fast_rcnn_reg_loss
        valid_losses[step, 4] = fast_rcnn_class_acc

        progbar.update(step + 1,
                       [('rpn_cls', rpn_class_loss),
                        ('rpn_regr', rpn_reg_loss),
                        ('detector_cls', fast_rcnn_class_loss),
                        ('detector_regr', fast_rcnn_reg_loss)])

        if step == n_valid_steps - 1 and C.verbose:
            mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor_valid)
                                            ) / len(rpn_accuracy_for_epoch_valid)
            rpn_accuracy_rpn_monitor_valid = []
            print(f'\nValidation: Average number of overlapping bounding boxes '
                  f'from RPN = {mean_overlapping_bboxes}')
            if mean_overlapping_bboxes == 0:
                print('RPN is not producing bounding boxes that overlap the ground truth boxes.'
                      ' Check RPN settings or keep training.')

            loss_rpn_cls = np.mean(valid_losses[:, 0])
            loss_rpn_regr = np.mean(valid_losses[:, 1])
            loss_class_cls = np.mean(valid_losses[:, 2])
            loss_class_regr = np.mean(valid_losses[:, 3])
            class_acc = np.mean(valid_losses[:, 4])

            mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch_valid)
                                            ) / len(rpn_accuracy_for_epoch_valid)
            rpn_accuracy_for_epoch_valid = []

            if C.verbose:
                print("Validation Metrics: ")
                print(
                    f'Mean number of bounding boxes from RPN overlapping '
                    f'ground truth boxes: {mean_overlapping_bboxes}')
                print(f'Classifier accuracy for bounding boxes from RPN: {class_acc}')
                print(f'Loss RPN classifier: {loss_rpn_cls}')
                print(f'Loss RPN regression: {loss_rpn_regr}')
                print(f'Loss Detector classifier: {loss_class_cls}')
                print(f'Loss Detector regression: {loss_class_regr}')
                print(f'Elapsed time: {time.time() - start_time}')

            curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
            print("Total validation loss: %.4f" % curr_loss)
            start_time = time.time()
            break

    val_classifier_metric.reset_states()
