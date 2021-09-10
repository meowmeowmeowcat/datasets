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

"""Utils to read envlogger data into a dataset that is compatible with RLDS."""

from typing import Any, Callable, Dict, Generator, List, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf


def get_episode_metadata(
    episode: Sequence[Any]) -> Dict[str, Any]:
  """Extracts the episode metadata and returns it as a dictionary.

  This is only used when the episode metadata is stored in the first step.

  Args:
    episode: sequence of steps (envlogger.StepData).

  Returns:
    Dictionary containing the episode metadata
  """

  first_step = episode[0]
  if not first_step.timestep.first():
    raise ValueError('The first step in this episode is not first')
  custom_data = first_step.custom_data
  return custom_data.get('episode_metadata', {}) if custom_data else {}


def get_step_metadata(step: Any) -> Dict[str, Any]:
  """Extracts the custom metadata from the step.

  Args:
    step: step data (envlogger.StepData).

  Returns:
    Dictionary containing the step metadata.
  """
  if not isinstance(step.custom_data, dict):
    # We ignore step metadata if step.custom_data is not a dictionary
    return {}
  return {k: v for k, v in step.custom_data.items()}


def get_episode_dict(episode: Sequence[Any],
                     episode_metadata: Dict[str, Any],
                     keep_episode_fn: Callable[[Dict[str, Any]],
                                               bool] = lambda x: True,
                     step_fn: Callable[[Dict[str, Any]],
                                       Dict[str, Any]] = lambda x: x,
                     episode_fn: Callable[[Dict[str, Any]],
                                          Dict[str, Any]] = lambda x: x,
                     steps_to_ds=False) -> Tuple[Dict[str, Any], bool]:
  """Obtains an RLDS episode from an Envlogger episode.

  Args:
    episode: sequence of envlogger.StepData.
    episode_metadata: metadata as obtained from the envlogger reader.
    keep_episode_fn: function to apply to the episode metadata to decide if the
      episode has to be kept.
    step_fn: functiont hat processes each RLDS step once it's built from the raw
      data.
    episode_fn: function that processes each RLDS episode once it's built from
      the raw data.
    steps_to_ds: if True, the steps in the episode are coverted to a
      `tf.data.Dataset`.

  Returns:
     Tuple of an episode and a boolean indicating if the episode is valid.
  """
  if episode_metadata:
    episode_dict = episode_metadata
  else:
    episode_dict = get_episode_metadata(episode)
  if not keep_episode_fn(episode_dict):
    return ({}, False)
  steps = _generate_steps(episode, step_fn)

  if steps_to_ds:
    steps = tf.data.Dataset.from_tensor_slices(steps)
  episode_dict['steps'] = steps
  episode_dict = episode_fn(episode_dict)
  return (episode_dict, True)


def generate_episodes(
    tag_reader: Any,
    keep_episode_fn: Callable[[Dict[str, Any]], bool] = lambda x: True,
    step_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = lambda x: x,
    episode_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = lambda x: x,
    steps_to_ds=False) -> Generator[Dict[str, Any], None, None]:
  """Generates episodes by reading them from the tag_reader.

  Args:
    tag_reader: Envlogger reader (envlogger.Reader).
    keep_episode_fn: function that receives an episode and returns a boolean,
      acting as a filter (true if we want to keep this episodes, false if we
      want to skip it).
    step_fn: function that gets an RLDS step and returns the processed step.
    episode_fn: function that gets the RLDS episode and returns the processed
      episode.
    steps_to_ds: if True, the steps of each episode are returned as a nested
      dataset.

  Yields:
    A dictionary representing one episode.
  """
  for index, episode_metadata in enumerate(tag_reader.episode_metadata()):
    episode = tag_reader.episodes[index]
    episode_dict, keep = get_episode_dict(episode, episode_metadata,
                                          keep_episode_fn, step_fn, episode_fn,
                                          steps_to_ds)
    if keep:
      yield episode_dict


def _build_empty_step(
    step: Any, step_fn: Callable[[Dict[str, Any]],
                                 Dict[str, Any]]) -> Dict[str, Any]:
  """Builds a step with the same shape and dtype of an RLDS step."""
  #  Although some of the fields main contain data, it should be used as a step
  # containing non-defined data to get the shape and the dtype of the RLDS step.
  rlds_step = get_step_metadata(step)
  rlds_step['observation'] = step.timestep.observation
  rlds_step['action'] = step.action
  rlds_step['reward'] = step.timestep.reward
  rlds_step['discount'] = step.timestep.discount
  rlds_step['is_terminal'] = False
  rlds_step['is_first'] = False
  rlds_step['is_last'] = False

  return step_fn(rlds_step)


def _build_step(
    prev_step: Any, step: Any,
    step_fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
  """Builds an RLDS step from two envlogger steps."""
  rlds_step = get_step_metadata(prev_step)
  rlds_step['observation'] = prev_step.timestep.observation
  rlds_step['action'] = step.action
  rlds_step['reward'] = step.timestep.reward
  rlds_step['discount'] = step.timestep.discount
  rlds_step['is_terminal'] = False
  rlds_step['is_first'] = prev_step.timestep.first()
  rlds_step['is_last'] = prev_step.timestep.last()

  return step_fn(rlds_step)


def _build_last_step(
    prev_step: Any, step_fn: Callable[[Dict[str, Any]],
                                      Dict[str, Any]]) -> Dict[str, Any]:
  """Builds the last RLDS step from an envlogger step."""
  # We append the observation of the final step (action and reward were
  # included in the previous step.
  # The terminal flag is inferred like in termination(), truncation()
  # from dm_env/_environment.py
  is_terminal = (
      prev_step.timestep.last() and prev_step.timestep.discount == 0.0)
  rlds_step = get_step_metadata(prev_step)
  rlds_step['observation'] = prev_step.timestep.observation
  rlds_step['is_terminal'] = is_terminal
  rlds_step['is_first'] = prev_step.timestep.first()
  rlds_step['is_last'] = True
  # Discount, action and reward are meaningless in the terminal step
  rlds_step['action'] = tf.nest.map_structure(np.zeros_like, prev_step.action)
  rlds_step['reward'] = np.zeros_like(prev_step.timestep.reward)
  rlds_step['discount'] = np.zeros_like(prev_step.timestep.discount)

  return step_fn(rlds_step)


def _generate_steps(
    episode: Sequence[Any],
    step_fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
  """Constructs a dictionary of steps for the given episode.

  Args:
    episode: Sequence of steps (envlogger.StepData). Assumes that there are at
      least two steps.
    step_fn: function that gets an RLDS step and returns the processed step.

  Returns:
    Nested dictionary with the steps of one episode.
  """
  if len(episode) < 2:
    step = _build_last_step(episode[0], step_fn)
    return _to_nested_list(step)
  empty_step = _build_empty_step(episode[1], step_fn)
  steps = {k: _empty_nested_list(v) for k, v in empty_step.items()}

  prev_step = None
  for step in episode:
    if prev_step is not None:
      rlds_step = _build_step(prev_step, step, step_fn)
      for k, v in rlds_step.items():
        steps[k] = _append_nested(steps[k], v)
    prev_step = step

  if prev_step is not None:
    last_step = _build_last_step(prev_step, step_fn)
    for k, v in last_step.items():
      steps[k] = _append_nested(steps[k], v)
  return steps


def _append_nested(
    collection: Any, data: Union[np.ndarray, List[Any], Tuple[Any, ...],
                                 Dict[str, Any]]) -> Any:
  """Appends data to the more nested dimension of collection.

  Args:
    collection: collection constructed by appending elements like data to the
      most nested dimension. It is of type Union[np.ndarray, List[Any],
      Tuple[Any, ...], Dict[str, Any]] but using this causes type errors because
      it considers that we are trying to append to a dict or a tuple.
    data: data logged by DM Env Logger that can be a scalar, array, dict or
      tuple.

  Returns:
    Collection with data appended to the nested dimension.
  """
  if isinstance(data, dict):
    for k in collection:
      _append_nested(collection[k], data[k])
    return collection
  elif isinstance(data, tuple):
    return tuple(
        _append_nested(collection[index], value)
        for index, value in enumerate(data))
  else:
    collection.append(data)
    return collection


def _empty_nested_list(
    data: Union[np.ndarray, List[Any], Tuple[Any, ...], Dict[str, Any]]
) -> Union[List[Any], Tuple[Any, ...], Dict[str, Any]]:
  """Creates an element like data but with empty lists in the nested dimensions.

  Args:
    data: scalar, array, list, tuple or dictionary.

  Returns:
    item with the same shape of data but with empty lists in its nested
    dimensions.
  """
  if isinstance(data, dict):
    return {k: _empty_nested_list(data[k]) for k in data}
  elif isinstance(data, tuple):
    return tuple(_empty_nested_list(x) for x in data)
  else:
    return []


def _to_nested_list(
    data: Union[np.ndarray, List[Any], Tuple[Any, ...], Dict[str, Any]]
) -> Union[List[Any], Tuple[Any, ...], Dict[str, Any]]:
  """Transforms data to lists of length 1 in the nested dimensions.

  Args:
    data: scalar, array, list, tuple or dictionary.

  Returns:
    item with the same shape of data but with lists of length 1 in its nested
    dimensions.
  """
  if isinstance(data, dict):
    return {k: _to_nested_list(data[k]) for k in data}
  elif isinstance(data, tuple):
    return tuple(_to_nested_list(x) for x in data)
  else:
    return [data]
