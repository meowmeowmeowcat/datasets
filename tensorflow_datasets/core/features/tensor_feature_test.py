# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors.
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

# coding=utf-8
"""Tests for tensorflow_datasets.core.features.tensor_feature."""

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_datasets import testing
from tensorflow_datasets.core import features as features_lib


class FeatureTensorTest(testing.FeatureExpectationsTestCase):

  def test_shape_static(self):

    np_input = np.random.rand(2, 3).astype(np.float32)
    array_input = [
        [1, 2, 3],
        [4, 5, 6],
    ]

    self.assertFeature(
        feature=features_lib.Tensor(shape=(2, 3), dtype=tf.float32),
        dtype=tf.float32,
        shape=(2, 3),
        tests=[
            # Np array
            testing.FeatureExpectationItem(
                value=np_input,
                expected=np_input,
            ),
            # Python array
            testing.FeatureExpectationItem(
                value=array_input,
                expected=array_input,
            ),
            # Invalid dtype
            testing.FeatureExpectationItem(
                # On Windows, np default dtype is `int32`
                value=np.random.randint(256, size=(2, 3), dtype=np.int64),
                raise_cls=ValueError,
                raise_msg='int64 do not match',
            ),
            # Invalid shape
            testing.FeatureExpectationItem(
                value=np.random.rand(2, 4).astype(np.float32),
                raise_cls=ValueError,
                raise_msg='are incompatible',
            ),
        ],
    )

  def test_shape_dynamic(self):

    np_input_dynamic_1 = np.random.randint(256, size=(2, 3, 2), dtype=np.int32)
    np_input_dynamic_2 = np.random.randint(256, size=(5, 3, 2), dtype=np.int32)

    self.assertFeature(
        feature=features_lib.Tensor(shape=(None, 3, 2), dtype=tf.int32),
        dtype=tf.int32,
        shape=(None, 3, 2),
        tests=[
            testing.FeatureExpectationItem(
                value=np_input_dynamic_1,
                expected=np_input_dynamic_1,
            ),
            testing.FeatureExpectationItem(
                value=np_input_dynamic_2,
                expected=np_input_dynamic_2,
            ),
            # Invalid shape
            testing.FeatureExpectationItem(
                value=np.random.randint(256, size=(2, 3, 1), dtype=np.int32),
                raise_cls=ValueError,
                raise_msg='are incompatible',
            ),
        ])

  def test_bool_flat(self):

    self.assertFeature(
        feature=features_lib.Tensor(shape=(), dtype=tf.bool),
        dtype=tf.bool,
        shape=(),
        tests=[
            testing.FeatureExpectationItem(
                value=np.array(True),
                expected=True,
            ),
            testing.FeatureExpectationItem(
                value=np.array(False),
                expected=False,
            ),
            testing.FeatureExpectationItem(
                value=True,
                expected=True,
            ),
            testing.FeatureExpectationItem(
                value=False,
                expected=False,
            ),
        ])

  def test_bool_array(self):

    self.assertFeature(
        feature=features_lib.Tensor(shape=(3,), dtype=tf.bool),
        dtype=tf.bool,
        shape=(3,),
        tests=[
            testing.FeatureExpectationItem(
                value=np.array([True, True, False]),
                expected=[True, True, False],
            ),
            testing.FeatureExpectationItem(
                value=[True, False, True],
                expected=[True, False, True],
            ),
        ])

  def test_string(self):
    nonunicode_text = 'hello world'
    unicode_text = u'你好'

    self.assertFeature(
        feature=features_lib.Tensor(shape=(), dtype=tf.string),
        shape=(),
        dtype=tf.string,
        tests=[
            # Non-unicode
            testing.FeatureExpectationItem(
                value=nonunicode_text,
                expected=tf.compat.as_bytes(nonunicode_text),
            ),
            # Unicode
            testing.FeatureExpectationItem(
                value=unicode_text,
                expected=tf.compat.as_bytes(unicode_text),
            ),
            # Empty string
            testing.FeatureExpectationItem(
                value='',
                expected=b'',
            ),
            # Trailing zeros
            testing.FeatureExpectationItem(
                value=b'abc\x00\x00',
                expected=b'abc\x00\x00',
            ),
        ],
    )

    self.assertFeature(
        feature=features_lib.Tensor(shape=(2, 1), dtype=tf.string),
        shape=(2, 1),
        dtype=tf.string,
        tests=[
            testing.FeatureExpectationItem(
                value=[[nonunicode_text], [unicode_text]],
                expected=[
                    [tf.compat.as_bytes(nonunicode_text)],
                    [tf.compat.as_bytes(unicode_text)],
                ],
            ),
            testing.FeatureExpectationItem(
                value=[nonunicode_text, unicode_text],  # Wrong shape
                raise_cls=ValueError,
                raise_msg='(2,) and (2, 1) must have the same rank',
            ),
            testing.FeatureExpectationItem(
                value=[['some text'], [123]],  # Wrong dtype
                raise_cls=TypeError,
                raise_msg='Expected binary or unicode string, got 123',
            ),
        ],
    )


if __name__ == '__main__':
  testing.test_main()