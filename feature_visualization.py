#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Author:AnRong

"""
import tensorflow as tf
import numpy as np
import PIL.Image as Image
import random, time
import os, cv2, glob, sys
import matplotlib.pyplot as plt
sys.path.append('../../')
from six.moves import xrange
#from i3d_nonlocal import InceptionI3d
from i3d import InceptionI3d
import saliency
print(tf.__version__)


# Basic model parameters as external flags.
flags = tf.app.flags
# gpu_num = 1
# flags.DEFINE_integer('batch_size', 1, 'Batch size.')
# flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('n_frames', 64, 'Nummber of frames per clib')
# flags.DEFINE_integer('n_objects', 10, 'Nummber of objects per frame')
# flags.DEFINE_integer('rgb_channels', 3, 'RGB_channels for input')
# flags.DEFINE_integer('classics', 51, 'The number of class')
# FLAGS = flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 64, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 24, 'The num of class')
FLAGS = flags.FLAGS


def data_resize(tmp_data, crop_size):
    img_datas = []
    for j in xrange(len(tmp_data)):
        height, width = tmp_data[j].shape[:2]
        if width > height:
            scale = float(crop_size) / float(height)
            img = cv2.resize(tmp_data[j], (int(width * scale + 1), crop_size))
        else:
            scale = float(crop_size) / float(width)
            img = cv2.resize(tmp_data[j], (int(high * scale + 1), crop_size))
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        img_datas.append(img)

    return img_datas


def get_data(filename, num_frames_per_clip=16, s_index=0):
    ret_arr = []
    filenames = ''
    for parent, dirnames, filenames in os.walk(filename):
        filenames = sorted(filenames)
        #print(filenames)
        if len(filenames)==0:
            print("Error, please check ...")

            return []
        if (len(filenames)-s_index) <= num_frames_per_clip:
            print("Not long enough, please check s_index ...")

            return []
        for i in range(num_frames_per_clip):
            image_name = str(filename) + '/' + str(filenames[i+s_index])
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)

    return ret_arr


def data_process(tmp_data, crop_size):
    img_datas = []
    for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(crop_size) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
            scale = float(crop_size) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        img_datas.append(img)

    return img_datas


def build_i3d_model(video_tensor):
    # model_name = "/home/ar/Experiment/ucf-101/rgb_backup01/models/rgb_scratch_10000_6_64_0.0001_decay/i3d_ucf_model-19999" # Note: I3D trained model
    model_name = "./models/rgb_imagenet_10000_6_64_0.0001_decay/i3d_ucf_model-9999"
    print("load model succeed")

    graph = tf.Graph()
    with graph.as_default():
        images_placeholder = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.n_frames, FLAGS.crop_size, FLAGS.crop_size, FLAGS.rgb_channels])
        #is_training = tf.placeholder(tf.bool)

        with tf.variable_scope('RGB'):
            logits, _ = InceptionI3d(
                           num_classes=FLAGS.classics,
                           spatial_squeeze=True,
                           final_endpoint='Logits', 
                           name='inception_i3d'
                           )(images_placeholder, is_training=False)

        # Create a saver for writing training checkpoints
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)

        # Restore trained model
        saver.restore(sess, model_name)

        neuron_selector = tf.placeholder(tf.int32)
        y = logits[0][neuron_selector]

        prediction = tf.argmax(logits, 1)

    out_feature = sess.run(logits, 
                           feed_dict={images_placeholder: video_tensor})

    prediction_class = sess.run(prediction, 
                                feed_dict={images_placeholder: video_tensor})[0]
    #print(prediction_class)

    ###############################################################################################
    #gradient_saliency = saliency.GradientSaliency(graph, sess, y, images_placeholder)

    # Compute the vanilla mask and the smoothed mask.
    #vanilla_mask_3d = gradient_saliency.GetMask(video_tensor[0], feed_dict = {neuron_selector: prediction_class})
    #print(vanilla_mask_3d.shape)
    #smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(video_tensor[0], feed_dict = {neuron_selector: prediction_class})

    #vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
    #print(vanilla_mask_grayscale.shape)
    #smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
    ###############################################################################################
    guided_backprop = saliency.GuidedBackprop(graph, sess, y, images_placeholder)

    # Compute the vanilla mask and the smoothed mask.
    vanilla_guided_backprop_mask_3d = guided_backprop.GetMask(video_tensor[0], feed_dict = {neuron_selector: prediction_class})
    smoothgrad_guided_backprop_mask_3d = guided_backprop.GetSmoothedMask(video_tensor[0], feed_dict = {neuron_selector: prediction_class})

    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_guided_backprop_mask_3d)
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_guided_backprop_mask_3d)
    ###############################################################################################

    return vanilla_mask_grayscale, smoothgrad_mask_grayscale


def vis_original(video_tensor):
    fig=plt.figure()

    for each_depth in range(16):
        fig.add_subplot(4,4,each_depth+1)
        plt.imshow(video_tensor[each_depth], cmap='jet')
    plt.show()


def vis_saliency(feature):   
    fig=plt.figure()

    for each_depth in range(16):
        fig.add_subplot(4,4,each_depth+1)
        plt.imshow(feature[each_depth], cmap='jet')
    plt.show()


def vis_feature(feature):
    fig=plt.figure()

    for each_depth in range(16):
        fig.add_subplot(4,4,each_depth+1)
        plt.imshow(feature[0,0,:,:,each_depth], cmap='jet')
    plt.show()


def vis_combine(video_tensor, feature):
    fig=plt.figure()
    depth1=0
    depth2=0

    for i in range(32): # Plot 32 fig.
        #print(depth1, depth2, i)
        if i%2 == 0:
            fig.add_subplot(8,4,i+1)
            plt.imshow(video_tensor[depth1], cmap='jet')
            depth1 += 1
        else:
            fig.add_subplot(8,4,i+1)
            plt.imshow(feature[depth2], cmap='jet')
            depth2 += 1
    plt.show()


if __name__ == '__main__':
    rgb_data = []
    rgb_ret_arr = get_data('/home/ee401_2/Documents/ucf_24/CricketBowling/v_CricketBowling_g16_c07/i', num_frames_per_clip=FLAGS.n_frames, s_index=10) # Frame data
    rgb_temp = data_process(rgb_ret_arr, FLAGS.crop_size)
    rgb_data.append(rgb_temp)
    video_tensor = np.array(rgb_data).astype(np.float32)
    f1, f2 = build_i3d_model(video_tensor)
    resize_ori = data_resize(rgb_ret_arr, FLAGS.crop_size)
    #vis_original(resize_ori)
    #vis_saliency(f2)
    vis_combine(resize_ori, f2)
