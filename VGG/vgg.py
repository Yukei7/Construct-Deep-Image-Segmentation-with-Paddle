import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import numpy as np
from basic_dataloader import BasicDataLoader
from basic_transforms import Compose, RandomScale, RandomFlip, Pad, RandomCrop

np.set_printoptions(precision=2)


class VGG16(fluid.dygraph.Layer):
    # VGG16: 13 convolution layers and 3 full connected layers.
    # input: 224 * 224 * 3
    # -> 2 x conv3-64
    # -> pooling
    # -> 2 x conv3-128
    # -> pooling
    # -> 3 x conv3-256
    # -> pooling
    # -> 3 x conv3-512
    # -> pooling
    # -> 3 x conv3-512
    # -> pooling
    # -> 2 x FC4096, 1 x FC1000 (drop-out)
    # -> softmax
    def __init__(self, x, num_classes):
        super(VGG16, self).__init__()
        self.x = x
        self.num_classes = num_classes

    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')


    def forward(self):
        conv1 = self.conv_block(self.x, 64, 2, [0.3, 0])
        conv2 = self.conv_block(conv1, 128, 2, [0.4, 0])
        conv3 = self.conv_block(conv2, 256, 3, [0.4, 0.4, 0])
        conv4 = self.conv_block(conv3, 512, 3, [0.4, 0.4, 0])
        conv5 = self.conv_block(conv4, 512, 3, [0.4, 0.4, 0])

        drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
        fc1 = fluid.layers.fc(input=drop, size=512, act=None)
        bn = fluid.layers.batch_norm(input=fc1, act='relu')
        
        drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
        fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
        predict = fluid.layers.fc(input=fc2, size=self.num_classes, act='softmax')
        return predict




def main():
    batch_size = 5
    place = fluid.CPUPlace()
    # place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        crop_size = 224
        transform = Compose([RandomScale(scales=[0.5, 1, 2]),
                             RandomFlip(prob=0.5),
                             Pad(size=crop_size),
                             RandomCrop(size=crop_size)])
        image_folder="./work/dummy_data"
        image_list_file="./work/dummy_data/list.txt"
        basic_dataloaer = BasicDataLoader(image_folder=image_folder, image_list_file=image_list_file, transform=transform, shuffle=True)
        dataloader = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)
        dataloader.set_sample_generator(basic_dataloaer, batch_size=batch_size, places=place)

        # model = VGG16(num_classes=2)

        # num_epoch = 5
        # for epoch in range(1, num_epoch+1):
        #     print(f'Epoch [{epoch}/{num_epoch}]:')
        #     for idx, (data, label) in enumerate(dataloader):
        #         image, label = transform(data, label)
        #         image = to_variable(image)
        #         label = to_variable(label)
        #         predict = model(image, label)
        #         cost = fluid.layers.mean_iou(input=image, label=label, num_classes=2)
        #         print(f'Iter {idx}, cost: {cost}')
        #     optimizer = fluid.optimizer.Adam(learning_rate=0.001)
        #     optimizer.minimize(avg_cost)

        #     exe = fluid.Executor(place)


if __name__ == "__main__":
    main()