# -- coding:utf-8 --
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=1000000, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=100, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()
fake_num=0
EPS = 1e-12
CROP_SIZE = 256
pool_size=50
Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

fake_images_AB = []
fake_images_BA = []
def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]#只有tensor可以这样做,第4维，获取通道数
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")#对padded进行填充,长宽各增加2
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")#四维输入和四维卷积核进行卷积
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)#复制x
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)#计算每个batch_size的图像所有元素求均值和方差
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv             #反卷积函数


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))#获取所有匹配的文件路径列表
    decode = tf.image.decode_jpeg #解码函数
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path)) #获取文件名
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path))) #若文件名为有数字则以数字排序
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train") #生成文件队列，若train模式则打乱
        reader = tf.WholeFileReader() #读取队列中的文件
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)#图像解码
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32) #转换图像格式

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels") #确认图像有3通道
        with tf.control_dependencies([assertion]):#执行依赖关系
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]
            a_images = preprocess(raw_input[:,:width//2,:])#a为左半边图像，b为右半边图像
            b_images = preprocess(raw_input[:,width//2:,:])

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size)) #每个batch中有多少个图像                               #按顺序组合一个batch

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, generator_outputs_channels,name="generator"):
    with tf.variable_scope(name):
        layers = []
    
        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = conv(generator_inputs, a.ngf, stride=2)
            layers.append(output)
    
        layer_specs = [
            a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]  #number of generator filters in first conv layer
            a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]
                   
        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):#解码部分总共七层
                rectified = lrelu(layers[-1], 0.2)#激活函数
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(rectified, out_channels, stride=2)
                output = batchnorm(convolved)
                layers.append(output)
    
        layer_specs = [
            (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2] #编码部分总共七层
            (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]
    
        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):#生成字典
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)#在某一维度两个tensor相连
    
                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = deconv(rectified, out_channels)
                output = batchnorm(output)
    
                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)
    
                layers.append(output)
    
        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = deconv(rectified, generator_outputs_channels)
            output = tf.tanh(output)
            layers.append(output)
    
        return layers[-1]

def fake_image_pool(num_fakes, fake, fake_pool):
        if(num_fakes < pool_size):
            fake_pool.append(fake)
            return fake
        else :
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0,pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else :
                return fake
def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator_all") as scope:
        out_channels_a2b = int(targets.get_shape()[-1])
        outputs_a2b_t = create_generator(inputs, out_channels_a2b,name="a2b") #G的生成的fake
        out_channels_b2a = int(inputs.get_shape()[-1])
        cyc_b2a=create_generator(outputs_a2b_t,out_channels_b2a,name="b2a")
        
        scope.reuse_variables()
        outputs_b2a_t = create_generator(targets, out_channels_b2a,name="b2a") #G的生成的fake
        cyc_a2b=create_generator(outputs_b2a_t,out_channels_a2b,name="a2b")
        
    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("fake_discriminator_b"):
        with tf.variable_scope("discriminator_b"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake_b_t = create_discriminator(inputs, outputs_a2b_t)#对假的进行判定
    with tf.name_scope("fake_discriminator_a"):
        with tf.variable_scope("discriminator_a"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake_a_t = create_discriminator(targets, outputs_b2a_t)#对假的进行判定
    
    with tf.name_scope("generator_loss_a2b"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN_a2b = tf.reduce_mean(-tf.log(predict_fake_b_t + EPS))#越小越好
        gen_loss_L1_a2b = tf.reduce_mean(tf.abs(inputs - cyc_b2a))
        gen_loss_a2b = gen_loss_GAN_a2b * a.gan_weight + gen_loss_L1_a2b * a.l1_weight#越小越好，G的loss
    
    with tf.name_scope("generator_loss_b2a"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN_b2a = tf.reduce_mean(-tf.log(predict_fake_a_t + EPS))#越小越好
        gen_loss_L1_b2a = tf.reduce_mean(tf.abs(targets - cyc_a2b))
        gen_loss_b2a = gen_loss_GAN_b2a * a.gan_weight + gen_loss_L1_b2a * a.l1_weight#越小越好，G的loss
    with tf.name_scope("real_discriminator_b"):
        with tf.variable_scope("discriminator_b",reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real_b_t = create_discriminator(inputs, targets) #对真实的进行判定
            
    with tf.name_scope("real_discriminator_a"):
        with tf.variable_scope("discriminator_a",reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real_a_t = create_discriminator(targets, inputs) #对真实的进行判定
    global fake_num
    fake_AB_sam=fake_image_pool(fake_num,outputs_a2b_t,fake_images_AB)
    fake_BA_sam=fake_image_pool(fake_num,outputs_b2a_t,fake_images_BA)
    fake_num=fake_num+1
    
    with tf.name_scope("fake_discriminator_b"):
        with tf.variable_scope("discriminator_b",reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake_b_t = create_discriminator(inputs,fake_AB_sam)#对假的进行判定
    with tf.name_scope("fake_discriminator_a"):
        with tf.variable_scope("discriminator_a",reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake_a_t = create_discriminator(targets, fake_BA_sam) #假的进行判定
        
    with tf.name_scope("discriminator_loss_a"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        d_loss_real=tf.reduce_mean(-(tf.log(predict_real_a_t + EPS)))
        d_loss_fake=tf.reduce_mean(tf.log(1 - predict_fake_a_t + EPS))
        discrim_loss_a = tf.reduce_mean(-(tf.log(predict_real_a_t + EPS) + tf.log(1 - predict_fake_a_t + EPS)))#越小越好，D_a的loss
    with tf.name_scope("discriminator_loss_b"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss_b = tf.reduce_mean(-(tf.log(predict_real_b_t + EPS) + tf.log(1 - predict_fake_b_t + EPS)))#越小越好，D_b的loss
    
#    with tf.name_scope("gan_single_train"):
#        gan_loss=gen_loss_a2b+gen_loss_b2a
#        gan_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
#        gan_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
#        gan_grads_and_vars = gan_optim.compute_gradients(gan_loss, var_list=gan_tvars)
#        gan_train = gan_optim.apply_gradients(gan_grads_and_vars)

   
        
    with tf.name_scope("discriminator_train"):
        discrim_loss=discrim_loss_a+discrim_loss_b
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]#判别器中所有可训练参数
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)#计算变量列表计算损失函数的梯度，返回梯度列表
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)#更新梯度对应参数的状态，把梯度应用到变量上面去，返回应用的操作
    with tf.name_scope("generator_train"):        
        gen_loss=gen_loss_a2b+gen_loss_b2a
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
        gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)#这个函数用于更新参数，就是采用滑动平均的方法更新参数,意义在于提高模型在测试数据上的健壮性
    gen_loss_GAN=gen_loss_GAN_a2b+gen_loss_GAN_b2a
    gen_loss_L1=gen_loss_L1_a2b+gen_loss_L1_b2a
    update_losses = ema.apply([gen_loss_GAN, gen_loss_L1])
    update_losses1= ema.apply([discrim_loss])
    #update_losses2=ema.apply([gen_loss_GAN, gen_loss_L1])
    global_step = tf.contrib.framework.get_or_create_global_step() #得到目前模型训练到达全局步数
       
    incr_global_step = tf.assign(global_step, global_step+1)#将global_step赋予新值，即步数加一
    
    #train1=tf.group(update_losses1)
    discrim_loss=ema.average(discrim_loss)
    d_loss_real=ema.average(d_loss_real)
    d_loss_fake=ema.average(d_loss_fake)
    discrim_grads_and_vars=discrim_grads_and_vars#返回判别模型的梯度
    gen_loss_GAN=ema.average(gen_loss_GAN)
    gen_loss_L1=ema.average(gen_loss_L1)
    gen_grads_and_vars=gen_grads_and_vars #返回生成模型的梯度
    outputs_a2b=outputs_a2b_t
    outputs_b2a=outputs_b2a_t
    predict_real_a=predict_real_a_t
    predict_fake_a=predict_fake_a_t
    predict_real_b=predict_real_b_t
    predict_fake_b=predict_fake_b_t 
    train=tf.group(update_losses,update_losses1, incr_global_step, gen_train)
    #train1=tf.group(gan_train)
    return train,discrim_loss,discrim_grads_and_vars,gen_loss_GAN,gen_loss_L1,gen_grads_and_vars,outputs_a2b,outputs_b2a,predict_real_a,predict_fake_a,predict_real_b,predict_fake_b

def save_images(fetches, step=None):#将输出图像保存到output_dir中,返回一个列表
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):#enumerate对列表返回索引和元素
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs_a2b", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs_a2b", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)#生成0到2**31 - 1的随机数

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():#打印a中的所有元素
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))#vars读取模块或对象的信息

    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")

        input = tf.placeholder(tf.string, shape=[1])
        input_data = tf.decode_base64(input[0])
        input_image = tf.image.decode_png(input_data)

        # remove alpha channel if present
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4), lambda: input_image[:,:,:3], lambda: input_image)
        # convert grayscale to RGB
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 1), lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)

        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image.set_shape([CROP_SIZE, CROP_SIZE, 3])
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator"):
            batch_output = deprocess(create_generator(preprocess(batch_input), 3))

        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        if a.output_filetype == "png":
            output_data = tf.image.encode_png(output_image)
        elif a.output_filetype == "jpeg":
            output_data = tf.image.encode_jpeg(output_image, quality=80)
        else:
            raise Exception("invalid filetype")
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key":  tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

        return

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    train,discrim_loss,discrim_grads_and_vars,gen_loss_GAN,gen_loss_L1,gen_grads_and_vars,outputs_a2b,outputs_b2a,predict_real_a,predict_fake_a,predict_real_b,predict_fake_b = create_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(outputs_a2b, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs_a2b = deprocess(outputs_a2b)
            outputs_b2a = deprocess(outputs_b2a)
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs_a2b = deprocess(outputs_a2b)
        outputs_b2a = deprocess(outputs_b2a)
    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs_a2b = convert(outputs_a2b)
        converted_outputs_b2a = convert(outputs_b2a)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),#使用函数作用于所有元素
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs_a2b": tf.map_fn(tf.image.encode_png, converted_outputs_a2b, dtype=tf.string, name="output_pngs_a2b"),
            "outputs_b2a": tf.map_fn(tf.image.encode_png, converted_outputs_b2a, dtype=tf.string, name="output_pngs_b2a"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary_a2b"):
        tf.summary.image("outputs_a2b", converted_outputs_a2b)

    with tf.name_scope("outputs_summary_b2a"):
        tf.summary.image("outputs_b2a", converted_outputs_b2a)

    with tf.name_scope("predict_real_summary_a"):
        tf.summary.image("predict_real_a", tf.image.convert_image_dtype(predict_real_a, dtype=tf.uint8))
    
    with tf.name_scope("predict_real_summary_b"):
        tf.summary.image("predict_real_b", tf.image.convert_image_dtype(predict_real_b, dtype=tf.uint8))
        
    with tf.name_scope("predict_fake_summary_a"):
        tf.summary.image("predict_fake_a", tf.image.convert_image_dtype(predict_fake_a, dtype=tf.uint8))
    
    #with tf.name_scope("predict_fake_summary_b"):
        #tf.summary.image("predict_fake_b", tf.image.convert_image_dtype(predict_fake_b, dtype=tf.uint8))
    print("2")
    tf.summary.scalar("discriminator_loss", discrim_loss)
    tf.summary.scalar("generator_loss_GAN", gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var) #参数变化

    for grad, var in discrim_grads_and_vars + gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad) #梯度变化

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])#tf.reduce_sum为在tensor所有维度上元素求和
                 #tf.reduce_prob为tensor所有维度元素求乘积
    saver = tf.train.Saver(max_to_keep=1)#保存最近的一个模型

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None) # 创建tf.train.Supervisor来管理模型的训练过程   
                                                                                # (a) 自动去 checkpoint 加载数据或者初始化数据
                                                                               #（b) 自动有一个 Saver ，可以用来保存 checkpoint
                                                                                  #  eg: sv.saver.save(sess, save_path)
    with sv.managed_session() as sess: #会自动取logdir中找checkpoint
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)#来自动获取最后一次保存的模型
            saver.restore(sess, checkpoint)

        max_steps = 2**32 #默认值
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)

            print("wrote index at", index_path)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)#配置运行时需要记录的信息
                    run_metadata = tf.RunMetadata()#tf.RunMetadata()定义元信息，对模型进行记录和保存 即运行时记录运行信息的proto

                fetches = {
                    "train": train,
                    "global_step": sv.global_step,
                }
                #fetches1={"train1":train1}
                #print("the step is",step)
                if should(a.progress_freq):
                    #fetches["discrim_loss"] = discrim_loss #给字典添加新的值
                    fetches["gen_loss_GAN"] = gen_loss_GAN
                    fetches["gen_loss_L1"] = gen_loss_L1
                    #fetches["predict_fake_b"]=tf.reduce_mean(predict_fake_b)
                    fetches["discrim_loss"]= discrim_loss
                   # fetches["d_loss_real"]=d_loss_real
                    #fetches["d_loss_fake"]=d_loss_fake
                    #fetches1["gen_loss_GAN"]=gen_loss_GAN
                    #fetches1["gen_loss_L1"]=gen_loss_L1
                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches
                #print("1")
                results = sess.run(fetches, options=options, run_metadata=run_metadata)
                
                
               # results1 = sess.run(fetches1, options=options, run_metadata=run_metadata)
                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])
                   # print("d_loss_real",results["d_loss_real"])
                    #print("d_loss_fake",results["d_loss_fake"])
                   # print("predict",results["predict_fake_b"])
                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()
