from torch import nn
from networks.models import Generator, Discriminator

from typing import List
import config


def build_asrnet_model() -> nn.Module:
    model = Generator(config.in_channels, config.out_channels, config.upscale_factor)
    model = model.to(device=config.device, non_blocking=True)

    return model


def build_asrgan_model() -> List[nn.Module]:
    discriminator = Discriminator()
    generator = Generator(
        config.in_channels, config.out_channels, config.upscale_factor
    )

    # Transfer to CUDA
    discriminator = discriminator.to(device=config.device)
    generator = generator.to(device=config.device)

    return discriminator, generator
