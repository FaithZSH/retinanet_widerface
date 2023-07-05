from __future__ import print_function
import os
import argparse

import skimage
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from utils.timer import Timer


labels = dict()
labels[0] = 'face'
img_dim = 640


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def normalizer(image):
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    return (image.astype(np.float32) - mean) / std


def unnormalizer(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def resizer(image, min_side=img_dim, max_side=img_dim):
    rows, cols, cns = image.shape
    smallest_side = min(rows, cols)
    scale = min_side / smallest_side
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
    rows, cols, cns = image.shape
    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32
    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    return torch.from_numpy(new_image), scale


def noResizer(image):
    rows, cols, cns = image.shape
    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32
    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)

    return torch.from_numpy(new_image)


def main():
    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str,
                        help='Dir to save txt results')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--dataset_folder', default='/data/data/code/zsh/widerface/WIDER_Face/WIDER_val/images/',
                        type=str, help='dataset path')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
    parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
    parser.add_argument('--trained_model', help='Path to pretrained model')
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    net = torch.load(args.trained_model)

    net.eval()
    print('Finished loading model!')
    # print(net)
    rgb_mean = (104, 117, 123)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "wider_val.txt"
    print(testset_list)

    with open(testset_list, 'r') as fr:
        test_img_path_list = fr.read().split()
    num_images = len(test_img_path_list)
    print(num_images)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i, img_name in enumerate(test_img_path_list):
        p = 1
        while img_name.split('_')[p][0] != img_name[0]:
            p += 1
        folds = ''
        ll = 0
        for x in range(p):
            ll += len(img_name.split('_')[x])
            folds += img_name.split('_')[x] + "_"
        fold = folds[:-1]
        img_name = img_name[ll + p:]
        image_path = testset_folder + fold + '/' + img_name + '.jpg'
        # print(image_path)

        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # b, g, r = cv2.split(img_raw)
        # # 以RGB的形式重新组合。
        # rgb_image = cv2.merge([r, g, b])
        img = np.float32(img_raw) - rgb_mean

        # img_raw_rgb = skimage.io.imread(image_path)
        #
        # img = img_raw_rgb.astype(np.float32)  # - rgb_mean
        img = img / 255.0

        img = normalizer(img)

        # 进行resize
        # img, scale = resizer(img)
        img = noResizer(img)
        scale = 1.0

        img = np.transpose(img, (2, 0, 1))
        data = torch.unsqueeze(img, dim=0)
        # print(data.shape)

        _t['forward_pass'].tic()

        scores, classification, transformed_anchors = net(data.cuda().float())

        _t['forward_pass'].toc()

        _t['misc'].tic()

        img = np.array(255 * unnormalizer(data[0, :, :, :])).copy()
        img[img < 0] = 0
        img[img > 255] = 255
        idxs = np.where(scores.cpu() > 0.5)

        _t['misc'].toc()
        save_name = args.save_folder + fold + '/' + img_name + ".txt"
        print(save_name)
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(idxs[0].shape)[1:-2] + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            scores = scores.cpu().tolist()
            if args.save_image:
                img = np.transpose(img, (1, 2, 0))
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                w = int(x2 - x1)
                h = int(y2 - y1)
                # print(x1, x2, y1, y2)

                confidence = str(scores[j])
                line = str(x1) + " " + str(y1) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)
                if args.save_image:
                    if not os.path.exists("./results/"):
                        os.makedirs("./results/")
                    draw_caption(img, (x1, y1, x2, y2), label_name)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                    name = "./results/" + str(i) + ".jpg"
                    cv2.imwrite(name, img)

            #     draw_caption(img, (x1, y1, x2, y2), label_name)
            #     cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            #
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images,
                                                                                     _t['forward_pass'].average_time,
                                                                                     _t['misc'].average_time))


if __name__ == '__main__':
    main()
