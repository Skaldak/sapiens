# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from mmengine.model import BaseModule
from torch import nn

from mmpretrain.registry import MODELS


@MODELS.register_module()
class CAELoss(BaseModule):
    """Loss function for CAE.

    Compute the align loss and the main loss.

    Args:
        lambd (float): The weight for the align loss.
    """

    def __init__(self, lambd: float) -> None:
        super().__init__()
        self.lambd = lambd
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()

    def forward(
            self, logits: torch.Tensor, target: torch.Tensor,
            latent_pred: torch.Tensor,
            latent_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function of CAE Loss.

        Args:
            logits (torch.Tensor): The outputs from the decoder.
            target (torch.Tensor): The targets generated by dalle.
            latent_pred (torch.Tensor): The latent prediction from the
                regressor.
            latent_target (torch.Tensor): The latent target from the teacher
                network.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The main loss and align loss.
        """
        loss_main = self.loss_cross_entropy(logits, target)
        loss_align = self.loss_mse(latent_pred,
                                   latent_target.detach()) * self.lambd

        return loss_main, loss_align
