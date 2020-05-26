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
"""PyTorch RoBERTa model. """

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
from transformers import BertPreTrainedModel, RobertaModel, RobertaConfig

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
}


class RobertaForSequenceClassificationConsistency(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassificationConsistency, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

        self.class_loss_fct = CrossEntropyLoss()
        self.consistency_loss_fct = L1Loss()

    # change the value of lambda
    def set_lambda(self, lambda_a, lambda_b):
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b

    def forward(self, input_ids, attention_mask, token_type_ids,
                aug_one_input_ids=None, aug_one_attention_mask=None, aug_one_token_type_ids=None,
                aug_two_input_ids=None, aug_two_attention_mask=None, aug_two_token_type_ids=None,
                position_ids=None, head_mask=None, labels=None,
                labels_one_hot=None, aug_labels_one_hot=None, paired=False, triplet=False):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=None,
                               position_ids=position_ids,
                               head_mask=head_mask)
        aug_one_outputs = self.roberta(aug_one_input_ids,
                                       attention_mask=aug_one_attention_mask,
                                       token_type_ids=None,
                                       position_ids=position_ids,
                                       head_mask=head_mask)
        aug_two_outputs = self.roberta(aug_two_input_ids,
                                       attention_mask=aug_two_attention_mask,
                                       token_type_ids=None,
                                       position_ids=position_ids,
                                       head_mask=head_mask)

        # pred for original data as usual
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]

        class_loss = self.class_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (class_loss, ) + outputs

        # pred for augmented data (first)
        aug_one_sequence_output = aug_one_outputs[0]
        aug_one_logits = self.classifier(aug_one_sequence_output)
        aug_one_outputs = (aug_one_logits,) + aug_one_outputs[2:]

        # pred for augmented data (second)
        aug_two_sequence_output = aug_two_outputs[0]
        aug_two_logits = self.classifier(aug_two_sequence_output)
        aug_two_outputs = (aug_two_logits,) + aug_two_outputs[2:]

        # calculate symmetric consistency loss
        orig_pred_target_probs = torch.sum(labels_one_hot * logits, dim=-1)
        aug_pred_target_probs = torch.sum(aug_labels_one_hot * aug_one_logits, dim=-1)
        paired_mask = (paired == 1).type(torch.FloatTensor).cuda()
        l_sym = paired_mask * self.consistency_loss_fct(orig_pred_target_probs, aug_pred_target_probs)
        l_sym = l_sym.mean()
        outputs = (l_sym, ) + outputs

        # Calculate transitive consistency loss
        l_trans_a = self.consistency_loss_fct(logits[:, 0] + aug_one_logits[:, 0],  aug_two_logits[:, 0])
        l_trans_b = self.consistency_loss_fct(logits[:, 0] + aug_one_logits[:, 1], aug_two_logits[:, 1])

        triplet_mask = (triplet == 1).type(torch.FloatTensor).cuda()
        l_trans = triplet_mask * (l_trans_a + l_trans_b)
        l_trans = l_trans.mean()
        outputs = (l_trans, ) + outputs
        
        # Calculate final loss
        loss = class_loss + self.lambda_a * l_sym + self.lambda_b * l_trans

        outputs = (loss, ) + outputs

        return outputs  # (loss), (consistency_loss), (class_loss), logits, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
