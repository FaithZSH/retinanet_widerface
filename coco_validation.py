import argparse
import torch
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import CocoDataset, NoResizer, Normalizer, Augmenter
from retinanet import coco_eval
from data import preproc
assert torch.__version__.split('.')[0] == '1'
from test_widerface import img_dim


print('CUDA available: {}'.format(torch.cuda.is_available()))
rgb_mean = (104, 117, 123)  # bgr order


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str)

    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                              transform=transforms.Compose([Normalizer(),
                                                            NoResizer()
                                                            ])
                              # , preproc=preproc(img_dim=img_dim, rgb_means=rgb_mean)
                              )

    print('Num training images: {}'.format(len(dataset_val)))

    # Create the model
    retinanet = torch.load(parser.model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    coco_eval.evaluate_coco(dataset_val, retinanet)


if __name__ == '__main__':
    main()
