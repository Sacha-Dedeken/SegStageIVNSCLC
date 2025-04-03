"""Monai Attention Unet with Instance Norm instead of Batch Norm."""

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from typing import Union

import torch
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm
from torch import nn

__all__ = ["AttentionUnet"]


class ConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        strides: int = 1,
        dropout: float = 0.0,
        norm_type: str = "INSTANCE",  # Add norm_type parameter
    ) -> None:
        super().__init__()
        layers = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=getattr(Norm, norm_type),
                dropout=dropout,
            ),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=getattr(Norm, norm_type),
                dropout=dropout,
            ),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c: torch.Tensor = self.conv(x)
        return x_c


class UpConv(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        strides: int = 2,
        dropout: float = 0.0,
        norm_type: str = "INSTANCE",  # Add norm_type parameter
    ) -> None:
        super().__init__()
        self.up = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act="relu",
            adn_ordering="NDA",
            norm=getattr(Norm, norm_type),
            dropout=dropout,
            is_transposed=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_u: torch.Tensor = self.up(x)
        return x_u


class AttentionBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        f_int: int,
        f_g: int,
        f_l: int,
        dropout: float = 0.0,
        norm_type: str = "INSTANCE",  # Add norm_type parameter
    ) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[getattr(Norm, norm_type), spatial_dims](
                f_int
            ),  # Use the norm_type correctly
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[getattr(Norm, norm_type), spatial_dims](
                f_int
            ),  # Use the norm_type correctly
        )

        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[getattr(Norm, norm_type), spatial_dims](
                1
            ),  # Use the norm_type correctly
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        submodule: nn.Module,
        dropout: float = 0.0,
        norm_type: str = "INSTANCE",  # Add norm_type parameter
    ) -> None:
        super().__init__()
        self.attention = AttentionBlock(
            spatial_dims=spatial_dims,
            f_g=in_channels,
            f_l=in_channels,
            f_int=in_channels // 2,
            norm_type=norm_type,  # Use norm_type
        )
        self.upconv = UpConv(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=in_channels,
            strides=2,
            norm_type=norm_type,  # Use norm_type
        )
        self.merge = Convolution(
            spatial_dims=spatial_dims,
            in_channels=2 * in_channels,
            out_channels=in_channels,
            dropout=dropout,
            norm=getattr(Norm, norm_type),  # Use norm_type
        )
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fromlower = self.upconv(self.submodule(x))
        att = self.attention(g=fromlower, x=x)
        att_m: torch.Tensor = self.merge(torch.cat((att, fromlower), dim=1))
        return att_m


class AttentionUnet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence,
        strides: Sequence,
        kernel_size: Union[Sequence, int] = 3,
        up_kernel_size: Union[Sequence, int] = 3,
        dropout: float = 0.0,
        norm_type: str = "INSTANCE",  # Add norm_type parameter
    ) -> None:
        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.dropout = dropout

        head = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            dropout=dropout,
            norm_type=norm_type,  # Use norm_type
        )
        reduce_channels = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            conv_only=True,
        )
        self.up_kernel_size = up_kernel_size

        def _create_block(
            channels: Sequence,
            strides: Sequence,
            level: int = 0,
        ) -> nn.Module:
            if len(channels) > 2:  # noqa: PLR2004
                subblock = _create_block(channels[1:], strides[1:], level=level + 1)
                return AttentionLayer(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=channels[1],
                    submodule=nn.Sequential(
                        ConvBlock(
                            spatial_dims=spatial_dims,
                            in_channels=channels[0],
                            out_channels=channels[1],
                            strides=strides[0],
                            dropout=self.dropout,
                            norm_type=norm_type,  # Use norm_type
                        ),
                        subblock,
                    ),
                    dropout=dropout,
                    norm_type=norm_type,  # Use norm_type
                )
            else:
                # the next layer is the bottom so stop recursion,
                # create the bottom layer as the subblock for this layer
                return self._get_bottom_layer(
                    channels[0],
                    channels[1],
                    strides[0],
                    norm_type,  # Pass norm_type
                )

        encdec = _create_block(self.channels, self.strides)
        self.model = nn.Sequential(head, encdec, reduce_channels)

    def _get_bottom_layer(
        self,
        in_channels: int,
        out_channels: int,
        strides: int,
        norm_type: str,  # Add norm_type parameter
    ) -> nn.Module:
        return AttentionLayer(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            submodule=ConvBlock(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                dropout=self.dropout,
                norm_type=norm_type,  # Use norm_type
            ),
            dropout=self.dropout,
            norm_type=norm_type,  # Use norm_type
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_m: torch.Tensor = self.model(x)
        return x_m
