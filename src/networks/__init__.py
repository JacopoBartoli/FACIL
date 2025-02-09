from torchvision import models

from .lenet import LeNet
from .vggnet import VggNet
from .resnet32 import resnet32
from .econv_vit import econv_vit, econv_vit_ext_attn
from .vit import vit, vit_small, vit_small_ext_attn
from .compact_convolutional_transformer import compact_convolutional_transformer, compact_convolutional_transformer_small, compact_convolutional_transformer_small_ext_attn

# available torchvision models
tvmodels = ['alexnet',
            'densenet121', 'densenet169', 'densenet201', 'densenet161',
            'googlenet',
            'inception_v3',
            'mobilenet_v2',
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
            'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
            'squeezenet1_0', 'squeezenet1_1',
            'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
            'wide_resnet50_2', 'wide_resnet101_2'
            ]

allmodels = tvmodels + ['resnet32', 'LeNet', 'VggNet', 
                        'compact_convolutional_transformer', 'compact_convolutional_transformer_small', 'compact_convolutional_transformer_small_ext_attn', 
                        'econv_vit', 'econv_vit_ext_attn',
                         'vit', 'vit_small', 'vit_small_ext_attn']


def set_tvmodel_head_var(model):
    if type(model) == models.AlexNet:
        model.head_var = 'classifier'
    elif type(model) == models.DenseNet:
        model.head_var = 'classifier'
    elif type(model) == models.Inception3:
        model.head_var = 'fc'
    elif type(model) == models.ResNet:
        model.head_var = 'fc'
    elif type(model) == models.VGG:
        model.head_var = 'classifier'
    elif type(model) == models.GoogLeNet:
        model.head_var = 'fc'
    elif type(model) == models.MobileNetV2:
        model.head_var = 'classifier'
    elif type(model) == models.ShuffleNetV2:
        model.head_var = 'fc'
    elif type(model) == models.SqueezeNet:
        model.head_var = 'classifier'
    else:
        raise ModuleNotFoundError
