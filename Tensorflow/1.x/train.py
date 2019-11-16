import tensorflow as tf
import numpy as np
import argparse

from networks.resnet_34 import ResNet
from input_fn import input_fn

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default='models/')
parser.add_argument("--data_dir", default='data/')

def train():
    tf.set_random_seed(111)

    model = ResNet(global_step=0, name="Res34"))

    train_inputs = input_fn(train_filenames, train_labels)
    eval_inputs = input_fn(eval_filenames, eval_labels)

    train_model_spec = ResNet.model_fn(train_inputs,)
    eval_model_spec = ResNet.model_fn(eval_inputs,)

    train_and_evaluate(train_model_spec, eval_model_spec, model_dir, params)

def train_and_evaluate(train_model_spec, eval_model_spec, model_dir, params):

    with tf.Session as sess:
        sess.run(train_model_spec['variable_init_op'])

        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)

        for epoch in range(params.num_epochs)

if if __name__ == "__main__":
    train()