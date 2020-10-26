import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
import numpy as np
import argparse
from utils import AverageMeter
from basic_model import BasicModel
from basic_dataloader import BasicDataLoader
from basic_seg_loss import Basic_SegLoss
from basic_data_preprocessing import TrainAugmentation, ValAugmentation

from fcn8s import FCN8s
from paddle.fluid.dygraph import to_variable

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='basic')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_folder', type=str, default='./work/dummy_data')
parser.add_argument('--train_image_list_file', type=str, default='./work/dummy_data/train_list.txt')
parser.add_argument('--val_image_list_file', type=str, default='./work/dummy_data/val_list.txt')
parser.add_argument('--checkpoint_folder', type=str, default='./work/output')
parser.add_argument('--save_freq', type=int, default=2)


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


def validation(dataloader, model, num_classes):
    # TODO: validation phase.
    model.eval()
    accuracies = []
    mious = []
    for image,label in dataloader():
        image = image[np.newaxis, :, :, :]
        label = label[np.newaxis, :, :, :]
        image = to_variable(image)
        image = fluid.layers.transpose(image, (0, 3, 1, 2))
        label = to_variable(label)

        pred = model(image)
        pred = fluid.layers.transpose(pred,perm=[0, 2, 3, 1])
        pred = fluid.layers.softmax(pred)
        pred_label = fluid.layers.argmax(pred,axis=3)

        pred_i = pred[0, :, :, :]
        pred_i = fluid.layers.reshape(pred_i, [-1, num_classes])
        label_i = label[0, :, :, 0]
        label_i = fluid.layers.reshape(label_i, [-1, 1])
        acc = fluid.layers.accuracy(input=pred_i, label=label_i)
        accuracies.append(acc.numpy())

        miou, _, _ = paddle.fluid.layers.mean_iou(pred_label, label, num_classes)
        mious.append(miou.numpy())
    print("[validation] accuracy/miou: {}/{}".format(np.mean(accuracies), np.mean(mious)))


def main():
    # Step 0: preparation
    place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        # Step 1: Define training dataloader
        train_augmentation = TrainAugmentation(image_size=256)
        train_basic_dataloader = BasicDataLoader(image_folder=args.image_folder, 
                                                 image_list_file=args.train_image_list_file,
                                                 transform=train_augmentation, 
                                                 shuffle=True)
        train_dataloader = fluid.io.DataLoader.from_generator(capacity=10, use_multiprocess=True)
        train_dataloader.set_sample_generator(train_basic_dataloader, 
                                              batch_size=args.batch_size,
                                              places=place)
        
        val_augmentation = ValAugmentation(image_size=256)
        val_basic_dataloader = BasicDataLoader(image_folder=args.image_folder, 
                                               image_list_file=args.val_image_list_file,
                                               transform=val_augmentation, 
                                               shuffle=False)

        total_batch = int(len(train_basic_dataloader) / args.batch_size)
        
        # Step 2: Create model
        if args.net == "basic":
            #TODO: create basicmodel
            model = BasicModel()
        elif args.net == "fcn8s":
            model = FCN8s()
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

            if epoch % args.save_freq == 0 or epoch == args.num_epochs:
                model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{train_loss}")

                # TODO: save model and optmizer states
                model_path = os.path.join(args.checkpoint_folder, f"{args.net}--Epoch-{epoch}-Loss-{train_loss}")
                model_dict = model.state_dict()
                fluid.save_dygraph(model_dict, model_path)
                optimizer_dict = optimizer.state_dict()
                fluid.save_dygraph(optimizer_dict, model_path)
                print(f'----- Save model: {model_path}.pdparams')
                print(f'----- Save optimizer: {model_path}.pdopt')

            # Validation
            validation(val_basic_dataloader, 
                       model, 
                       num_classes=59)
            




if __name__ == "__main__":
    main()
