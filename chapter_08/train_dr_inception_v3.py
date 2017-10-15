# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary to train Inception on the ImageNet data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

sys.path.append(os.path.realpath('..'))
import data_utils

# The gpus to use for this process
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"

import tensorflow as tf

from chapter_08 import inception_train
from chapter_08.dr_data import DRData

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_dir', "/home/ubuntu/vmshare/d/datasets/diabetic",
                           """Directory where to write event logs """
                           """and checkpoint.""")

def main(_):
    data_utils.prepare_dr_dataset(root_dir=FLAGS.dataset_dir, save_space=False)
    dataset = DRData(subset=FLAGS.subset)
    assert dataset.data_files()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    inception_train.train(dataset)


if __name__ == '__main__':
    tf.app.run()
