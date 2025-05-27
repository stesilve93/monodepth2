# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class RangeDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(RangeDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(512*400, 1024)
        self.linear2 = nn.Linear(1024, 1)
        self.Relu = nn.ReLU()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        x = torch.reshape(x, (4,512*400))
        x = self.linear(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        self.outputs = self.Relu(x)


        return self.outputs
