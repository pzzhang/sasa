from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .smooth_resnet import *


class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias)

    def forward(self, x):
        return self.fc1(x.view(x.size(0), -1))