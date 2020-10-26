import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
import numpy as np
import argparse
from utils import AverageMeter

from data_loader import BasicDataLoader
from seg_loss import Basic_SegLoss

from data_preprocessing import Augmentation

from pspnet import PSPNet
from unet import UNet
from paddle.fluid.dygraph import to_variable

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='unet')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--image_folder', type=str, default='./work/dummy_data')
parser.add_argument('--image_list_file', type=str, default='./work/dummy_data/list.txt')
parser.add_argument('--checkpoint_folder', type=str, default='./work/output')
parser.add_argument('--save_freq', type=int, default=25)
parser.add_argument('--do_checkpoint', type=bool, default=False)


args = parser.parse_args()

np.set_printoptions(precision=3)

def train(dataloader, model, criterion, optimizer, epoch, total_batch):
    model.train()
    train_loss_meter = AverageMeter()
    for batch_id, data in enumerate(dataloader):
        #TODO:
        image = data[0]
        label = data[1]
        image = fluid.layers.transpose(image, (0, 3, 1, 2))
        pred = model(image)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.minimize(loss)
        model.clear_gradients()

        n = image.shape[0]
        train_loss_meter.update(loss.numpy()[0], n)
        print(f"Epoch[{epoch:03d}/{args.num_epochs:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"Average Loss: {train_loss_meter.avg:4f}")

    return train_loss_meter.avg


def validation(dataloader, val_size, model, num_classes):
    # TODO: validation phase.
    model.eval()
    accuracies = []
    mious = []
    counter = 0
    for image,label in dataloader():
        counter += 1
        image = image.astype(np.float32)
        label = label.astype(np.int64)
        image = image[np.newaxis, :, :, :]
        label = label[np.newaxis, :, :, :]

        image = to_variable(image)
        label = to_variable(label)

        image = fluid.layers.transpose(image, perm=[0, 3, 1, 2])
        pred = model(image)
        pred = fluid.layers.softmax(pred, axis=1)
        pred_label = fluid.layers.argmax(pred,axis=1)
        # NCHW -> NHWC
        pred = fluid.layers.transpose(pred,perm=[0, 2, 3, 1])

        pred_i = pred[0, :, :, :]
        pred_i = fluid.layers.reshape(pred_i, [-1, num_classes])
        label_i = label[0, :, :, 0]
        label_i = fluid.layers.reshape(label_i, [-1, 1])
        acc = fluid.layers.accuracy(input=pred_i, label=label_i)
        accuracies.append(acc.numpy())

        miou, _, _ = paddle.fluid.layers.mean_iou(pred_label, label, num_classes)
        mious.append(miou.numpy())
        if counter > val_size:
            break
    print("[validation] accuracy/miou: {}/{}".format(np.mean(accuracies), np.mean(mious)))


def main():
    # Step 0: preparation
    place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        # Step 1: Define training dataloader
        train_augmentation = Augmentation(image_size=256)
        train_basic_dataloader = BasicDataLoader(image_folder=args.image_folder, 
                                                 image_list_file=args.image_list_file,
                                                 transform=train_augmentation, 
                                                 shuffle=True)
        train_dataloader = fluid.io.DataLoader.from_generator(capacity=10, use_multiprocess=True)
        train_dataloader.set_sample_generator(train_basic_dataloader, 
                                              batch_size=args.batch_size,
                                              places=place)

        val_dataloader = BasicDataLoader(image_folder=args.image_folder, 
                                         image_list_file=args.image_list_file,
                                         transform=None,
                                         shuffle=True)
        

        total_batch = int(len(train_basic_dataloader) / args.batch_size)
        
        # Step 2: Create model
        if args.net == "pspnet":
            model = PSPNet()
        elif args.net == "unet":
            model = UNet()
        else:
            raise NotImplementedError(f"args.net: {args.net} is not Supported!")

        # Step 3: Define criterion and optimizer
        criterion = Basic_SegLoss
        optimizer = AdamOptimizer(learning_rate=args.lr, 
                                  parameter_list=model.parameters())

        # create optimizer
        
        # Step 4: Training
        for epoch in range(1, args.num_epochs+1):
            train_loss = train(train_dataloader,
                               model,
                               criterion,
                               optimizer,
                               epoch,
                               total_batch)
            print(f"----- Epoch[{epoch}/{args.num_epochs}] Train Loss: {train_loss:.4f}")

            if args.do_checkpoint and (epoch % args.save_freq == 0 or epoch == args.num_epochs):
                # TODO: save model and optmizer states
                model_path = os.path.join(args.checkpoint_folder, f"{args.net}--Epoch-{epoch}-Loss-{train_loss}")
                model_dict = model.state_dict()
                fluid.save_dygraph(model_dict, model_path)
                optimizer_dict = optimizer.state_dict()
                fluid.save_dygraph(optimizer_dict, model_path)
                print(f'----- Save model: {model_path}.pdparams')
                print(f'----- Save optimizer: {model_path}.pdopt')

            # Validation
            validation(val_dataloader,
                       val_size = 5,
                       model = model, 
                       num_classes=59)
        
        model_path = os.path.join(args.checkpoint_folder, f"{args.net}")
        model_dict = model.state_dict()
        fluid.save_dygraph(model_dict, model_path)
        optimizer_dict = optimizer.state_dict()
        fluid.save_dygraph(optimizer_dict, model_path)
        print(f'----- Save model: {model_path}.pdparams')
        print(f'----- Save optimizer: {model_path}.pdopt')
            




if __name__ == "__main__":
    main()