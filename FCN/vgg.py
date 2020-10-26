import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Dropout
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Linear

model_path = {
        #'vgg16': './vgg16',
        'vgg16bn': './vgg16_bn',
        # 'vgg19': './vgg19',
        # 'vgg19bn': './vgg19_bn'
        }

class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 use_bn=True,
                 act='relu',
                 name=None):
        super(ConvBNLayer, self).__init__(name)

        self.use_bn = use_bn
        if use_bn:
            self.conv = Conv2D(num_channels=num_channels,
                                num_filters=num_filters,
                                filter_size=filter_size,
                                stride=stride,
                                padding=(filter_size-1)//2,
                                groups=groups,
                                act=None,
                                bias_attr=None)
            self.bn = BatchNorm(num_filters, act=act)
        else:
            self.conv = Conv2D(num_channels=num_channels,
                                num_filters=num_filters,
                                filter_size=filter_size,
                                stride=stride,
                                padding=(filter_size-1)//2,
                                groups=groups,
                                act=act,
                                bias_attr=None)

    def forward(self, inputs):
        y = self.conv(inputs)
        if self.use_bn:
            y = self.bn(y)
        return y



class VGG(fluid.dygraph.Layer):
    def __init__(self, layers=16, use_bn=False, num_classes=1000):
        super(VGG, self).__init__()
        self.layers = layers
        self.use_bn = use_bn
        supported_layers = [16, 19]
        assert layers in supported_layers

        if layers == 16:
            depth = [2, 2, 3, 3, 3]
        elif layers == 19:
            depth = [2, 2, 4, 4, 4]

        num_channels = [3, 64, 128, 256, 512]
        num_filters = [64, 128, 256, 512, 512]

        self.layer1 = fluid.dygraph.Sequential(*self.make_layer(num_channels[0], num_filters[0], depth[0], use_bn, name='layer1'))
        self.layer2 = fluid.dygraph.Sequential(*self.make_layer(num_channels[1], num_filters[1], depth[1], use_bn, name='layer2'))
        self.layer3 = fluid.dygraph.Sequential(*self.make_layer(num_channels[2], num_filters[2], depth[2], use_bn, name='layer3'))
        self.layer4 = fluid.dygraph.Sequential(*self.make_layer(num_channels[3], num_filters[3], depth[3], use_bn, name='layer4'))
        self.layer5 = fluid.dygraph.Sequential(*self.make_layer(num_channels[4], num_filters[4], depth[4], use_bn, name='layer5'))

        self.classifier = fluid.dygraph.Sequential(
                Linear(input_dim=512 * 7 * 7, output_dim=4096, act='relu'),
                Dropout(),
                Linear(input_dim=4096, output_dim=4096, act='relu'),
                Dropout(),
                Linear(input_dim=4096, output_dim=num_classes))
                
        self.out_dim = 512 * 7 * 7


    def forward(self, inputs):
        x = self.layer1(inputs)
        x = fluid.layers.pool2d(x, pool_size=2, pool_stride=2)
        x = self.layer2(x)
        x = fluid.layers.pool2d(x, pool_size=2, pool_stride=2)
        x = self.layer3(x)
        x = fluid.layers.pool2d(x, pool_size=2, pool_stride=2)
        x = self.layer4(x)
        x = fluid.layers.pool2d(x, pool_size=2, pool_stride=2)
        x = self.layer5(x)
        x = fluid.layers.pool2d(x, pool_size=2, pool_stride=2)
        x = fluid.layers.adaptive_pool2d(x, pool_size=(7,7), pool_type='avg')
        x = fluid.layers.reshape(x, shape=[-1, self.out_dim])
        x = self.classifier(x)

        return x

    def make_layer(self, num_channels, num_filters, depth, use_bn, name=None):
        layers = []
        layers.append(ConvBNLayer(num_channels, num_filters, use_bn=use_bn, name=f'{name}.0'))
        for i in range(1, depth):
            layers.append(ConvBNLayer(num_filters, num_filters, use_bn=use_bn, name=f'{name}.{i}'))
        return layers


def VGG16(pretrained=False):
    model = VGG(layers=16)
    if pretrained:
        model_dict, _ = fluid.load_dygraph(model_path['vgg16'])
        model.set_dict(model_dict)
    return model

def VGG16BN(pretrained=False):
    model = VGG(layers=16, use_bn=True)
    if pretrained:
        model_dict, _ = fluid.load_dygraph(model_path['vgg16bn'])
        model.set_dict(model_dict)
    return model

# def VGG19(pretrained=False):
#     model =  VGG(layers=19)
#     if pretrained:
#         model_dict, _ = fluid.load_dygraph(model_path['vgg19'])
#         model.set_dict(model_dict)
#     return model

# def VGG19BN(pretrained=False):
#     model =  VGG(layers=19, use_bn=True)
#     if pretrained:
#         model_dict, _ = fluid.load_dygraph(model_path['vgg19bn'])
#         model.set_dict(model_dict)
#     return model



def main():
    with fluid.dygraph.guard():
        x_data = np.random.rand(2, 3, 224, 224).astype(np.float32)
        x = to_variable(x_data)

        # model = VGG16()
        # model.eval()
        # pred = model(x)
        # print('vgg16: pred.shape = ', pred.shape)

        model = VGG16BN()
        model.eval()
        pred = model(x)
        print('vgg16bn: pred.shape = ', pred.shape)

        # model = VGG19()
        # model.eval()
        # pred = model(x)
        # print('vgg19: pred.shape = ', pred.shape)

        # model = VGG19BN()
        # model.eval()
        # pred = model(x)
        # print('vgg19bn: pred.shape = ', pred.shape)

if __name__ == "__main__":
    main()
