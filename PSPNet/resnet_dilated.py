import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Linear

model_path = {'ResNet18': './resnet18',
              'ResNet34': './resnet34',
              'ResNet50': './resnet50',
              'ResNet101': './resnet101',
              'ResNet152': './resnet152'}

class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 dilation=1,
                 padding=None,
                 name=None):
        super(ConvBNLayer, self).__init__(name)

        if padding is None:
            padding = (filter_size-1)//2
        else:
            padding=padding

        self.conv = Conv2D(num_channels=num_channels,
                            num_filters=num_filters,
                            filter_size=filter_size,
                            stride=stride,
                            padding=padding,
                            groups=groups,
                            act=None,
                            dilation=dilation,
                            bias_attr=False)
        self.bn = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.bn(y)
        return y


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1  # expand ratio for last conv output channel in each block
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 shortcut=True,
                 name=None):
        super(BasicBlock, self).__init__(name)
        
        self.conv0 = ConvBNLayer(num_channels=num_channels,
                                 num_filters=num_filters,
                                 filter_size=3,
                                 stride=stride,
                                 act='relu',
                                 name=name)
        self.conv1 = ConvBNLayer(num_channels=num_filters,
                                 num_filters=num_filters,
                                 filter_size=3,
                                 act=None,
                                 name=name)
        if not shortcut:
            self.short = ConvBNLayer(num_channels=num_channels,
                                     num_filters=num_filters,
                                     filter_size=1,
                                     stride=stride,
                                     act=None,
                                     name=name)
        self.shortcut = shortcut

    def forward(self, inputs):
        conv0 = self.conv0(inputs)
        conv1 = self.conv1(conv0)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = fluid.layers.elementwise_add(x=short, y=conv1, act='relu')
        return y


class BottleneckBlock(fluid.dygraph.Layer):
    expansion = 4
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 shortcut=True,
                 dilation=1,
                 padding=None,
                 name=None):
        super(BottleneckBlock, self).__init__(name)

        self.conv0 = ConvBNLayer(num_channels=num_channels,
                                 num_filters=num_filters,
                                 filter_size=1,
                                 act='relu')
#                                 name=name)
        self.conv1 = ConvBNLayer(num_channels=num_filters,
                                 num_filters=num_filters,
                                 filter_size=3,
                                 stride=stride,
                                 padding=padding,
                                 act='relu',
                                 dilation=dilation)
 #                                name=name)
        self.conv2 = ConvBNLayer(num_channels=num_filters,
                                 num_filters=num_filters * 4,
                                 filter_size=1,
                                 stride=1)
 #                                name=name)
        if not shortcut:
            self.short = ConvBNLayer(num_channels=num_channels,
                                     num_filters=num_filters * 4,
                                     filter_size=1,
                                     stride=stride)
#                                     name=name)
        self.shortcut = shortcut
        self.num_channel_out = num_filters * 4

    def forward(self, inputs):
        conv0 = self.conv0(inputs)
        #print('conv0 shape=',conv0.shape)
        conv1 = self.conv1(conv0)
        #print('conv1 shape=', conv1.shape)
        conv2 = self.conv2(conv1)
        #print('conv2 shape=', conv2.shape)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        #print('short shape=', short.shape)
        y = fluid.layers.elementwise_add(x=short, y=conv2, act='relu')
        return y


class ResNet(fluid.dygraph.Layer):
    def __init__(self, layers=50, num_classes=1000):
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152]
        assert layers in supported_layers

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34:
            depth = [3, 4, 6, 3]
        elif layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]

        if layers < 50:
            num_channels = [64, 64, 128, 256]
        else:
            num_channels = [64, 256, 512, 1024]

        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(num_channels=3,
                                num_filters=64,
                                filter_size=7,
                                stride=2,
                                act='relu')
        self.pool2d_max = Pool2D(pool_size=3,
                                 pool_stride=2,
                                 pool_padding=1,
                                 pool_type='max')
        if layers < 50:
            block = BasicBlock
            l1_shortcut=True
        else:
            block = BottleneckBlock
            l1_shortcut=False
        
        self.layer1 = fluid.dygraph.Sequential(
                *self.make_layer(block,
                                 num_channels[0],
                                 num_filters[0],
                                 depth[0],
                                 stride=1,
                                 shortcut=l1_shortcut,
                                 name='layer1'))
        self.layer2 = fluid.dygraph.Sequential(
                *self.make_layer(block,
                                 num_channels[1],
                                 num_filters[1],
                                 depth[1],
                                 stride=2,
                                 name='layer2'))
        self.layer3 = fluid.dygraph.Sequential(
                *self.make_layer(block,
                                 num_channels[2],
                                 num_filters[2],
                                 depth[2],
                                 stride=1,
                                 name='layer3',
                                 dilation=2))
        self.layer4 = fluid.dygraph.Sequential(
                *self.make_layer(block,
                                 num_channels[3],
                                 num_filters[3],
                                 depth[3],
                                 stride=1,
                                 name='layer4',
                                 dilation=4))
        self.last_pool = Pool2D(pool_size=7, # ignore if global_pooling is True
                                global_pooling=True,
                                pool_type='avg')
        self.fc = Linear(input_dim=num_filters[-1] * block.expansion,
                         output_dim=num_classes,
                         act=None)

        self.out_dim = num_filters[-1] * block.expansion


    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool2d_max(x)

        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)

        x = self.last_pool(x)
        x = fluid.layers.reshape(x, shape=[-1, self.out_dim])
        x = self.fc(x)

        return x

    def make_layer(self, block, num_channels, num_filters, depth, stride, dilation=1, shortcut=False, name=None):
        layers = []
        if dilation > 1:
            padding=dilation
        else:
            padding=None

        layers.append(block(num_channels,
                            num_filters,
                            stride=stride,
                            shortcut=shortcut,
                            dilation=dilation,
                            padding=padding,
                            name=f'{name}.0'))
        for i in range(1, depth):
            layers.append(block(num_filters * block.expansion,
                                num_filters,
                                stride=1,
                                dilation=dilation,
                                padding=padding,
                                name=f'{name}.{i}'))
        return layers


def ResNet18(pretrained=False):
    model = ResNet(layers=18)
    if pretrained:
        model_state, _ = fluid.load_dygraph(model_path['ResNet18'])
        model.set_dict(model_state)
    return model


def ResNet34(pretrained=False):
    model =  ResNet(layers=34)
    if pretrained:
        model_state, _ = fluid.load_dygraph(model_path['ResNet34'])
        model.set_dict(model_state)
    return model


def ResNet50(pretrained=False):
    model =  ResNet(layers=50)
    if pretrained:
        model_state, _ = fluid.load_dygraph(model_path['ResNet50'])
        model.set_dict(model_state)
    return model


def ResNet101(pretrained=False):
    model = ResNet(layers=101)
    if pretrained:
        model_state, _ = fluid.load_dygraph(model_path['ResNet101'])
        model.set_dict(model_state)
    return model


def ResNet152(pretrained=False):
    model = ResNet(layers=152)
    if pretrained:
        model_state, _ = fluid.load_dygraph(model_path['ResNet152'])
        model.set_dict(model_state)
    return model

def main():
    with fluid.dygraph.guard():
        x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
        #x_data = np.random.rand(2, 3, 224, 224).astype(np.float32)
        x = to_variable(x_data)

        #model = ResNet18()
        #model.eval()
        #pred = model(x)
        #print('resnet18: pred.shape = ', pred.shape)

        #model = ResNet34()
        #pred = model(x)
        #model.eval()
        #print('resnet34: pred.shape = ', pred.shape)

        model = ResNet50()
        model.eval()
        pred = model(x)
        print('dilated resnet50: pred.shape = ', pred.shape)
        
        #model = ResNet101()
        #pred = model(x)
        #model.eval()
        #print('resnet101: pred.shape = ', pred.shape)

        #model = ResNet152()
        #pred = model(x)
        #model.eval()
        #print('resnet152: pred.shape = ', pred.shape)

        #print(model.sublayers())
        #for name, sub in model.named_sublayers(include_sublayers=True):
        #    #print(sub.full_name())
        #    if (len(sub.named_sublayers()))
        #    print(name)


if __name__ == "__main__":
    main()
