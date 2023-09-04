import torch
import torch.nn as nn
from compressai.layers import GDN as GeneralizedDivisiveNormalization
from torch import Tensor

def _conv( cin, cout, kernel_size, stride=1) -> nn.Conv2d:
    return nn.Conv2d(
        cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
    )

def _deconv(cin, cout, kernel_size, stride = 1) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        cin,
        cout,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class _AbsoluteValue(nn.Module):
    def forward(self, inp: Tensor) -> Tensor:
        return torch.abs(inp)
    
    
class ImageAnalysis(nn.Module):

    def __init__(self, network_channels: int, compression_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            _conv(3, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels),
            _conv(network_channels, compression_channels, kernel_size=5, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
class ImageSynthesis(nn.Module): 
    def __init__(self, network_channels: int, compression_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            _deconv(compression_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            _deconv(network_channels, 3, kernel_size=5, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
class HyperAnalysis(nn.Module):

    def __init__(self, network_channels: int, compression_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            _AbsoluteValue(),
            _conv(compression_channels, network_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
class HyperSynthesis(nn.Module):

    def __init__(self, network_channels: int, compression_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            _deconv(network_channels, compression_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)