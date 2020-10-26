from PIL import Image
import argparse
import os
from data_loader import BasicDataLoader
from pspnet import PSPNet
from unet import UNet
from data_preprocessing import Augmentation
import paddle.fluid as fluid
import cv2
import numpy as np
import paddle
from paddle.fluid.dygraph import to_variable

parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='./work/output')
parser.add_argument('--image_folder', type=str, default='./work/dummy_data')
parser.add_argument('--image_list_file', type=str, default='./work/dummy_data/list.txt')
parser.add_argument('--save_folder', type=str, default='./work/pred')
parser.add_argument('--method', type=str, default='resize')
parser.add_argument('--net', type=str, default='unet')

args = parser.parse_args()
np.set_printoptions(precision=3)


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

# def save_blend_image(image_file, pred_file):
#     image1 = Image.open(image_file)
#     image2 = Image.open(pred_file)
#     image1 = image1.convert('RGBA')
#     image2 = image2.convert('RGBA')
#     image = Image.blend(image1, image2, 0.5)
#     o_file = pred_file[0:-4] + "_blend.png"
#     image.save(o_file)


# def inference_resize(im, pred):
#     h, w = im.shape[0], im.shape[1]
#     pred = cv2.resize(pred, (w, h), cv2.INTER_NEAREST)
#     return pred

# def inference_sliding()

# def inference_multi_scale()



def save_images(im, pred, counter):
    palette = []
    with open('./work/color_files/pascal_context_colors.txt', 'r') as file:
        while True:
            line = file.readline().strip()
            if not line:
                break
            r, g, b = line.split(' ')[0], line.split(' ')[1], line.split(' ')[2]
            palette.extend([int(r), int(g), int(b)])
    color = colorize(pred, palette)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    color = color.convert('RGB')
    color.save(args.save_folder+'/'+str(counter)+'_pred.jpg')

    im = (im[0,:,:,:] * 255).astype(np.uint8)
    im = Image.fromarray(im)
    im = im.convert('RGB')
    im.save(args.save_folder+'/'+str(counter)+'.jpg')
    blend = Image.blend(im, color, 0.5)
    blend.save(args.save_folder+'/'+str(counter)+'_blend.jpg')



# this inference code reads a list of image path, and do prediction for each image one by one
def main():
    # 0. env preparation
    place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
    # 1. create model
        if args.net == "pspnet":
            model = PSPNet()
        elif args.net == "unet":
            model = UNet()
        else:
            raise NotImplementedError(f"args.net: {args.net} is not Supported!")
    # 2. load pretrained model
        para_state_dict, _ = fluid.load_dygraph(args.model_folder + '/' + args.net)
        model.set_dict(para_state_dict)
        model.eval()
    # 3. read test image list
    # 4. create transforms for test image, transform should be same as training
        train_augmentation = Augmentation(image_size=256)
        dataloader = BasicDataLoader(image_folder=args.image_folder, 
                                     image_list_file=args.image_list_file,
                                     transform=train_augmentation, 
                                     shuffle=False)

    # 5. loop over list of images
        counter = 0
        for im, _ in dataloader():
        # 6. read image and do preprocessing
            # 7. image to variable
            counter += 1
            im = im[np.newaxis, :, :, :]
            im = to_variable(im)
            # NHWC -> NCHW
            im = fluid.layers.transpose(im, (0, 3, 1, 2))
            pred = model(im)
            pred = fluid.layers.softmax(pred,axis=1)
            pred = fluid.layers.argmax(pred,axis=1)
            pred = pred.numpy()
            pred = np.squeeze(pred).astype('uint8')

            # 8. call inference func
            # if args.method == 'resize':
            #     pred = inference_resize(im, pred)
            # elif args.method == 'sliding':
            #     pass
            # else:
            #     raise Exception("Unexpected method '{}'".format(args.method))

            # 9. save results
            # NCHW -> NHWC
            im = fluid.layers.transpose(im, (0, 2, 3, 1))
            save_images(im.numpy(), pred, counter)


if __name__ == "__main__":
    main()
