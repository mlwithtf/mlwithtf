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
"""A binary to evaluate Inception on the DR data set.

Note that using the supplied pre-trained inception checkpoint, the eval should
achieve:
  precision @ 1 = N/A recall @ 5 = N/A [5126 examples]

See the README.md for more details.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
sys.path.append(os.path.realpath('../..'))
from book_code.data_utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import tensorflow as tf

from book_code.chapter_08 import inception_eval
from book_code.chapter_08.dr_data import DRData

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=None):
    prepare_dr_dataset(save_space=False)
    dataset = DRData(subset=FLAGS.subset)
    assert dataset.data_files()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    inception_eval.evaluate(dataset)


if __name__ == '__main__':
    tf.app.run()
