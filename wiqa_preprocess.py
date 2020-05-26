# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" WIQA processors and helpers """

import logging
import jsonlines
import os
from tqdm import tqdm
import re
import copy
import json
import random

from transformers.data.processors.utils import DataProcessor, InputExample

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label, example_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.example_id = example_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TripletInputFeatures(object):
    """
    A single set of features of triple data including three examples.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
        aug_one_input_ids: Indices of a paired augmented input tokens in the vocabulary.
        aug_one_attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        aug_one_token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        aug_two_input_ids: Indices of input sequence tokens in the vocabulary.
        aug_two_attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        aug_two_token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
        example_id: the example 
        labels_one_hot: the label of the original example represented as a one-hot vector. 
        aug_labels_one_hot: the label of the augmented example represented as a one-hot vector. 
        paired: 1 if the triple is a symmetric example, otherwise 0.
        triplet 1 if the triple is a transitive example, otherwise 0.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids,
                 aug_one_input_ids, aug_one_attention_mask, aug_one_token_type_ids,
                 aug_two_input_ids, aug_two_attention_mask, aug_two_token_type_ids,
                 label, example_id,
                 labels_one_hot, aug_labels_one_hot,
                 paired, triplet):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.aug_one_input_ids = aug_one_input_ids
        self.aug_one_attention_mask = aug_one_attention_mask
        self.aug_one_token_type_ids = aug_one_token_type_ids
        self.aug_two_input_ids = aug_two_input_ids
        self.aug_two_attention_mask = aug_two_attention_mask
        self.aug_two_token_type_ids = aug_two_token_type_ids
        self.label = label
        self.example_id = example_id
        self.labels_one_hot = labels_one_hot
        self.aug_labels_one_hot = aug_labels_one_hot
        self.paired = paired
        self.triplet = paired

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def multi_qa_convert_examples_to_features(examples, tokenizer,
                                          max_length=512,
                                          task=None,
                                          label_list=None,
                                          output_mode=None,
                                          pad_on_left=False,
                                          pad_token=0,
                                          pad_token_segment_id=0,
                                          mask_padding_with_zero=True):
    """
    Loads a data file into a list of `InputBatch`s
    """
    if task is not None:
        processor = multi_qa_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = multi_qa_output_modes[task]
            logger.info("Using output mode %s for task %s" %
                        (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        guid = example.guid
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1]
                              * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] *
                              padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + \
                ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + \
                ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" %
                        " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" %
                        " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label,
                          example_id=guid))

    return features


def multi_qa_triplet_convert_examples_to_features_augmented_data(examples, tokenizer,
                                                                 max_length=512,
                                                                 task=None,
                                                                 label_list=None,
                                                                 output_mode=None,
                                                                 pad_on_left=False,
                                                                 pad_token=0,
                                                                 pad_token_segment_id=0,
                                                                 mask_padding_with_zero=True,
                                                                 random_sample=False,
                                                                 active_sample=False,
                                                                 model=None,
                                                                 no_augmentation=True,
                                                                 random_augment_ratio=0.5):
    """
    Loads a data file into a list of `InputBatch`s
    """
    if task is not None:
        processor = multi_qa_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = multi_qa_output_modes[task]
            logger.info("Using output mode %s for task %s" %
                        (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}
    # create a qid2examples table.
    qid2examples = {example.guid: example for example in examples}

    # Create consistent pairs list.
    original_q_to_consistent_pairs = {}
    for example in examples:
        guid = example.guid
        if "_symmetric_" in guid:
            original_guid = guid.split("_symmetric_")[0]
            original_q_to_consistent_pairs.setdefault(
                original_guid, {"symmetric": [], "transitive": []})
            original_q_to_consistent_pairs[original_guid]["symmetric"].append(
                guid)
        elif "_transit_" in guid:
            original_guid = guid.split("@")[0]
            original_q_to_consistent_pairs.setdefault(
                original_guid, {"symmetric": [], "transitive": []})
            original_q_to_consistent_pairs[original_guid]["transitive"].append(
                guid)
        else:
            original_q_to_consistent_pairs.setdefault(
                guid, {"symmetric": [], "transitive": []})

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if "_symmetric_" in example.guid or "_transit_" in example.guid: 
            continue
        # add original examples
        of = _convert_example_to_features(example, tokenizer=tokenizer,
                                          label_map=label_map,
                                          max_length=max_length,
                                          task=task,
                                          label_list=label_list,
                                          output_mode=output_mode,
                                          pad_on_left=pad_on_left,
                                          pad_token=pad_token,
                                          pad_token_segment_id=pad_token_segment_id,
                                          mask_padding_with_zero=mask_padding_with_zero)
        if example.text_a is None:
            continue
        # load symmetric examples
        symm_examples = [qid2examples[q_id]
                         for q_id in original_q_to_consistent_pairs[example.guid]["symmetric"]]
        random.shuffle(symm_examples)
        labels_one_hot = [
            1 if i == of.label else 0 for i in range(len(label_list))]
        # Convert symmetric examples into features.
        for sym_i, sym_example in enumerate(symm_examples):
            sym_example_feature = _convert_example_to_features(sym_example, tokenizer=tokenizer,
                                                               label_map=label_map,
                                                               max_length=max_length,
                                                               task=task,
                                                               label_list=label_list,
                                                               output_mode=output_mode,
                                                               pad_on_left=pad_on_left,
                                                               pad_token=pad_token,
                                                               pad_token_segment_id=pad_token_segment_id,
                                                               mask_padding_with_zero=mask_padding_with_zero)

            aug_labels_one_hot = [
                1 if i == sym_example_feature.label else 0 for i in range(len(label_list))]

            # When multiple X_aug are generated from X, we shuffle the symmetric examples, we add (X, X_Aug^i) once, and add (X_aug^i, X) symmetric pairs for each X_Aug.
            # symmetric examples are shuffled not to overly represent the first symmetric examples.
            if sym_i == 0:
                features.append(TripletInputFeatures(input_ids=of.input_ids, attention_mask=of.attention_mask, token_type_ids=of.token_type_ids,
                                                     aug_one_input_ids=sym_example_feature.input_ids,
                                                     aug_one_attention_mask=sym_example_feature.attention_mask,
                                                     aug_one_token_type_ids=sym_example_feature.token_type_ids,
                                                     aug_two_input_ids=sym_example_feature.input_ids,
                                                     aug_two_attention_mask=sym_example_feature.attention_mask,
                                                     aug_two_token_type_ids=sym_example_feature.token_type_ids,
                                                     label=of.label, example_id=sym_example_feature.example_id,
                                                     labels_one_hot=labels_one_hot,
                                                     aug_labels_one_hot=aug_labels_one_hot,
                                                     paired=1, triplet=0))

            features.append(TripletInputFeatures(input_ids=sym_example_feature.input_ids, attention_mask=sym_example_feature.attention_mask,
                                                 token_type_ids=sym_example_feature.token_type_ids,
                                                 aug_one_input_ids=of.input_ids,
                                                 aug_one_attention_mask=of.attention_mask,
                                                 aug_one_token_type_ids=of.token_type_ids,
                                                 aug_two_input_ids=of.input_ids,
                                                 aug_two_attention_mask=of.attention_mask,
                                                 aug_two_token_type_ids=of.token_type_ids,
                                                 label=sym_example_feature.label, example_id=sym_example_feature.example_id,
                                                 labels_one_hot=aug_labels_one_hot,
                                                 aug_labels_one_hot=labels_one_hot,
                                                 paired=1, triplet=0))

        if len(symm_examples) == 0:
            # if there are no augmented results, add the same data into augmented data fields.
            # add dummy input
            features.append(TripletInputFeatures(input_ids=of.input_ids, attention_mask=of.attention_mask, token_type_ids=of.token_type_ids,
                                                 aug_one_input_ids=of.input_ids,
                                                 aug_one_attention_mask=of.attention_mask,
                                                 aug_one_token_type_ids=of.token_type_ids,
                                                 aug_two_input_ids=of.input_ids,
                                                 aug_two_attention_mask=of.attention_mask,
                                                 aug_two_token_type_ids=of.token_type_ids,
                                                 label=of.label, example_id=of.example_id,
                                                 labels_one_hot=labels_one_hot,
                                                 aug_labels_one_hot=labels_one_hot,
                                                 paired=0, triplet=0))

        # load symmetric examples
        transitive_examples = [qid2examples[q_id]
                               for q_id in original_q_to_consistent_pairs[example.guid]["transitive"]]
        for transit_i, transitive_example in enumerate(transitive_examples):
            x1_guid, x2_guid = transitive_example.guid.split(
                "@")[0],  transitive_example.guid.split("@")[1].split("_transit_")[0]
            x2_example = qid2examples[x2_guid]

            # Add transitive example
            x2_example_feature = _convert_example_to_features(x2_example, tokenizer=tokenizer,
                                                              label_map=label_map,
                                                              max_length=max_length,
                                                              task=task,
                                                              label_list=label_list,
                                                              output_mode=output_mode,
                                                              pad_on_left=pad_on_left,
                                                              pad_token=pad_token,
                                                              pad_token_segment_id=pad_token_segment_id,
                                                              mask_padding_with_zero=mask_padding_with_zero)
            x_trans_example_feature = _convert_example_to_features(transitive_example, tokenizer=tokenizer,
                                                                   label_map=label_map,
                                                                   max_length=max_length,
                                                                   task=task,
                                                                   label_list=label_list,
                                                                   output_mode=output_mode,
                                                                   pad_on_left=pad_on_left,
                                                                   pad_token=pad_token,
                                                                   pad_token_segment_id=pad_token_segment_id,
                                                                   mask_padding_with_zero=mask_padding_with_zero)
            # add original example to the task target once.
            if transit_i == 0:
                features.append(TripletInputFeatures(input_ids=of.input_ids,
                                                     attention_mask=of.attention_mask,
                                                     token_type_ids=of.token_type_ids,
                                                     aug_one_input_ids=x2_example_feature.input_ids,
                                                     aug_one_attention_mask=x2_example_feature.attention_mask,
                                                     aug_one_token_type_ids=x2_example_feature.token_type_ids,
                                                     aug_two_input_ids=x_trans_example_feature.input_ids,
                                                     aug_two_attention_mask=x_trans_example_feature.attention_mask,
                                                     aug_two_token_type_ids=x_trans_example_feature.token_type_ids,
                                                     label=of.label, example_id=x_trans_example_feature.example_id,
                                                     # make sure that the tiplet version never calculate irrelevant loss.
                                                     labels_one_hot=labels_one_hot, aug_labels_one_hot=labels_one_hot, \
                                                     paired=0, triplet=1))
            # add transitive examples (X2, X1, X_trans)
            features.append(TripletInputFeatures(input_ids=x2_example_feature.input_ids, attention_mask=x2_example_feature.attention_mask,
                                                 token_type_ids=x2_example_feature.token_type_ids,
                                                 aug_one_input_ids=of.input_ids,
                                                 aug_one_attention_mask=of.attention_mask,
                                                 aug_one_token_type_ids=of.token_type_ids,
                                                 aug_two_input_ids=x_trans_example_feature.input_ids,
                                                 aug_two_attention_mask=x_trans_example_feature.attention_mask,
                                                 aug_two_token_type_ids=x_trans_example_feature.token_type_ids,
                                                 label=x2_example_feature.label, example_id=x2_example_feature.example_id,
                                                 # make sure that the tiplet version never calculate irrelevant loss.
                                                 labels_one_hot=labels_one_hot, aug_labels_one_hot=labels_one_hot, \
                                                 paired=0, triplet=1))
            # Add transitive example as an independent examples.
            features.append(TripletInputFeatures(input_ids=x_trans_example_feature.input_ids, attention_mask=x_trans_example_feature.attention_mask,
                                                 token_type_ids=x_trans_example_feature.token_type_ids,
                                                 aug_one_input_ids=x_trans_example_feature.input_ids,
                                                 aug_one_attention_mask=x_trans_example_feature.attention_mask,
                                                 aug_one_token_type_ids=x_trans_example_feature.token_type_ids,
                                                 aug_two_input_ids=x_trans_example_feature.input_ids,
                                                 aug_two_attention_mask=x_trans_example_feature.attention_mask,
                                                 aug_two_token_type_ids=x_trans_example_feature.token_type_ids,
                                                 label=x_trans_example_feature.label, example_id=x_trans_example_feature.example_id,
                                                 labels_one_hot=labels_one_hot, aug_labels_one_hot=labels_one_hot, \
                                                 paired=0, triplet=0))

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in of.input_ids]))
            logger.info("attention_mask: %s" %
                        " ".join([str(x) for x in of.attention_mask]))
            logger.info("token_type_ids: %s" %
                        " ".join([str(x) for x in of.token_type_ids]))
            logger.info("label: %s " % (example.label))
    return features


def _convert_example_to_features(example, tokenizer, label_map,
                                 max_length=512,
                                 task=None,
                                 label_list=None,
                                 output_mode=None,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    guid = example.guid
    inputs = tokenizer.encode_plus(
        example.text_a,
        example.text_b,
        add_special_tokens=True,
        max_length=max_length
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1]
                          * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] *
                          padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + \
            ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + \
            ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
        len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
        len(attention_mask), max_length)
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
        len(token_type_ids), max_length)

    if output_mode == "classification":
        label = label_map[example.label]
    elif output_mode == "regression":
        label = float(example.label)
    else:
        raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids,
                         label=label,
                         example_id=guid)


class WIQAProcessor(DataProcessor):
    """Processor for the WIQA data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        return ["more", "less", "no_effect"]

    def _read_jsonl(self, jsonl_file):
        lines = []
        print("loading examples from {0}".format(jsonl_file))
        with jsonlines.open(jsonl_file) as reader:
            for obj in reader:
                lines.append(obj)
        return lines

    def _create_examples(self, lines, set_type, add_consistency=True):
        """Creates examples for the training and dev sets."""
        examples = []

        for (_, data_raw) in tqdm(enumerate(lines)):
            question = data_raw["question"]["stem"]
            para_steps = " ".join(data_raw["question"]['para_steps'])
            answer_labels = data_raw["question"]["answer_label"]
            example_id = data_raw['metadata']['ques_id']
            examples.append(
                InputExample(
                    guid=example_id,
                    text_a=question,
                    text_b=para_steps,
                    label=answer_labels))
        return examples


multi_qa_tasks_num_labels = {
    "wiqa": 3,
}

multi_qa_processors = {
    "wiqa": WIQAProcessor,
}

multi_qa_output_modes = {
    "wiqa": "classification",
}
