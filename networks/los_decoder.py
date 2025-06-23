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


class LOSDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(LOSDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.sigmoid = nn.Sigmoid()
        self.linear = None  # Initialize linear layer as None
        #self.linear = nn.Linear(524288, 1024)
        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 3)
        self.Relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, input_features):
        self.outputs = {}
            # Print all shapes in input_features
        # for i, feat in enumerate(input_features):
        #     print(f"Shape of input_features[{i}]: {feat.shape}")
        x = input_features[-1]               # Deepest encoder feature
        x = x.view(x.size(0), -1)            # Flatten to (B, C*H*W)

        # Define the first linear layer dynamically
        if self.linear is None:
            in_features = x.shape[1]
            self.linear = nn.Linear(in_features, 2048).to(x.device)

        x = self.linear(x)
        x = self.Relu(x)
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.Relu(x)
        x = self.linear3(x)
        x = self.Relu(x)

        self.outputs = x
        return self.outputs