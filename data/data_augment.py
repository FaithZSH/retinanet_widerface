import math

import cv2
import numpy as np
import random
import random as random_random
import torch

from utils.box_utils import matrix_iof


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    '''Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    '''
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class RandomBaiduCrop(object):
    '''Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    '''

    def __init__(self, size):

        self.mean = np.array([104, 117, 123], dtype=np.float32)
        self.maxSize = 12000  # max size
        self.infDistance = 9999999
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        random_counter = 0
        boxArea = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        # argsort = np.argsort(boxArea)
        # rand_idx = random.randint(min(len(argsort),6))
        # print('rand idx',rand_idx)
        rand_idx = random.randrange(len(boxArea))
        rand_Side = boxArea[rand_idx] ** 0.5
        # rand_Side = min(boxes[rand_idx,2] - boxes[rand_idx,0] + 1, boxes[rand_idx,3] - boxes[rand_idx,1] + 1)
        anchors = [16, 32, 64, 128, 256, 512]
        distance = self.infDistance
        anchor_idx = 5
        for i, anchor in enumerate(anchors):
            if abs(anchor - rand_Side) < distance:
                distance = abs(anchor - rand_Side)
                anchor_idx = i
        target_anchor = random.choice(anchors[0:min(anchor_idx + 1, 5) + 1])
        ratio = float(target_anchor) / rand_Side
        ratio = ratio * (2 ** random.uniform(-1, 1))
        if int(height * ratio * width * ratio) > self.maxSize * self.maxSize:
            ratio = (self.maxSize * self.maxSize / (height * width)) ** 0.5
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)
        image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)
        boxes[:, 0] *= ratio
        boxes[:, 1] *= ratio
        boxes[:, 2] *= ratio
        boxes[:, 3] *= ratio
        height, width, _ = image.shape
        sample_boxes = []
        xmin = boxes[rand_idx, 0]
        ymin = boxes[rand_idx, 1]
        bw = (boxes[rand_idx, 2] - boxes[rand_idx, 0] + 1)
        bh = (boxes[rand_idx, 3] - boxes[rand_idx, 1] + 1)
        w = h = self.size

        for _ in range(50):
            if w < max(height, width):
                if bw <= w:
                    w_off = random.uniform(xmin + bw - w, xmin)
                else:
                    w_off = random.uniform(xmin, xmin + bw - w)
                if bh <= h:
                    h_off = random.uniform(ymin + bh - h, ymin)
                else:
                    h_off = random.uniform(ymin, ymin + bh - h)
            else:
                w_off = random.uniform(width - w, 0)
                h_off = random.uniform(height - h, 0)
            w_off = math.floor(w_off)
            h_off = math.floor(h_off)
            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(w_off), int(h_off), int(w_off + w), int(h_off + h)])
            # keep overlap with gt box IF center in sampled patch
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            # mask in all gt boxes that above and to the left of centers
            m1 = (rect[0] <= boxes[:, 0]) * (rect[1] <= boxes[:, 1])
            # mask in all gt boxes that under and to the right of centers
            m2 = (rect[2] >= boxes[:, 2]) * (rect[3] >= boxes[:, 3])
            # mask in that both m1 and m2 are true
            mask = m1 * m2
            overlap = jaccard_numpy(boxes, rect)
            # have any valid boxes? try again if not
            if not mask.any() and not overlap.max() > 0.7:
                continue
            else:
                sample_boxes.append(rect)

        if len(sample_boxes) > 0:
            choice_idx = random.randrange(len(sample_boxes))
            choice_box = sample_boxes[choice_idx]
            # print('crop the box :',choice_box)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            m1 = (choice_box[0] < centers[:, 0]) * (choice_box[1] < centers[:, 1])
            m2 = (choice_box[2] > centers[:, 0]) * (choice_box[3] > centers[:, 1])
            mask = m1 * m2
            current_boxes = boxes[mask, :].copy()
            current_labels = labels[mask]
            current_boxes[:, :2] -= choice_box[:2]
            current_boxes[:, 2:] -= choice_box[:2]
            if choice_box[0] < 0 or choice_box[1] < 0:
                new_img_width = width if choice_box[0] >= 0 else width - choice_box[0]
                new_img_height = height if choice_box[1] >= 0 else height - choice_box[1]
                image_pad = np.zeros((new_img_height, new_img_width, 3), dtype=float)
                image_pad[:, :, :] = self.mean
                start_left = 0 if choice_box[0] >= 0 else -choice_box[0]
                start_top = 0 if choice_box[1] >= 0 else -choice_box[1]
                image_pad[start_top:, start_left:, :] = image

                choice_box_w = choice_box[2] - choice_box[0]
                choice_box_h = choice_box[3] - choice_box[1]

                start_left = choice_box[0] if choice_box[0] >= 0 else 0
                start_top = choice_box[1] if choice_box[1] >= 0 else 0
                end_right = start_left + choice_box_w
                end_bottom = start_top + choice_box_h
                current_image = image_pad[start_top:end_bottom, start_left:end_right, :].copy()
                return current_image, current_boxes, current_labels
            current_image = image[choice_box[1]:choice_box[3], choice_box[0]:choice_box[2], :].copy()
            return current_image, current_boxes, current_labels
        else:
            return image, boxes, labels


class RandomCrop(object):

    def __init__(self, image_size):
        self.options = [None, 0.3, 0.5, 0.7, 0.9]
        self.small_threshold = 8.0

    def __call__(self, im, boxes, labels):

        imh, imw, _ = im.shape
        short_size = min(imw, imh)
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)

        while True:
            mode = random_random.choice(self.options)
            for _ in range(50):
                if mode is None or mode < 0.7:
                    if mode is None:
                        w = short_size
                    else:
                        w = random_random.randrange(int(0.3 * short_size), int(1 * short_size))
                    h = w
                    if imw > w:
                        x = random_random.randrange(imw - w)
                    else:
                        x = 0
                    if imh > h:
                        y = random_random.randrange(imh - h)
                    else:
                        y = 0
                else:  # average sample
                    random_box = random_random.choice(boxes)
                    rbminx = random_box[0]
                    rbminy = random_box[1]
                    rbmaxx = random_box[2]
                    rbmaxy = random_box[3]
                    rbcx = (rbminx + rbmaxx) / 2
                    rbcy = (rbminy + rbmaxy) / 2
                    rbw = rbmaxx - rbminx
                    rbh = rbmaxy - rbminy
                    w = np.sqrt(rbw * rbh)
                    if w > 256:
                        random_scale = 640 / random_random.choice([64, 128, 256, 512])
                    elif w > 128:
                        random_scale = 640 / random_random.choice([32, 64, 128, 256])
                    elif w > 64:
                        random_scale = 640 / random_random.choice([16, 32, 64, 128])
                    elif w > 32:
                        random_scale = 640 / random_random.choice([16, 32, 64])
                    else:
                        random_scale = 640 / random_random.choice([16, 32])

                    w = int(w * random_scale)
                    h = w
                    _min_x = max(0, rbminx - max(w, rbw + 1) + rbw)
                    _min_y = max(0, rbminy - max(h, rbh + 1) + rbh)
                    if _min_x == rbminx:
                        x = _min_x
                    else:
                        x = random_random.randrange(_min_x, rbminx)
                    if _min_y == rbminy:
                        y = _min_y
                    else:
                        y = random_random.randrange(_min_y, rbminy)
                    roi_max_x = min(imw, x + w)
                    roi_max_y = min(imh, y + h)
                    w = roi_max_x - x
                    h = roi_max_y - y

                roi = torch.FloatTensor([[x, y, x + w, y + h]])
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                roi2 = roi.expand(len(center), 4)

                mask = (center > roi2[:, :2]) & (center < roi2[:, 2:])
                mask = mask[:, 0] & mask[:, 1]
                if not mask.any():
                    continue

                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
                img = im[y:y + h, x:x + w, :]

                selected_boxes[:, 0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:, 1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:, 2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:, 3].add_(-y).clamp_(min=0, max=h)

                boxes_uniform = selected_boxes / torch.FloatTensor([w, h, w, h]).expand_as(selected_boxes)
                boxwh = boxes_uniform[:, 2:] - boxes_uniform[:, :2]
                # mask = (boxwh[:,0] > self.small_threshold) & (boxwh[:,1] > self.small_threshold)
                mask = (boxwh[:, 0] * w * boxwh[:, 1] * h > self.small_threshold * self.small_threshold)
                if not mask.any():
                    continue

                selected_boxes_selected = selected_boxes.index_select(0, mask.nonzero().squeeze(1))
                selected_labels = labels.index_select(0, mask.nonzero().squeeze(1))
                # print(img.shape, 'shape')
                return img, selected_boxes_selected.numpy(), selected_labels.numpy()

            self.small_threshold = self.small_threshold / 2


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

    return image, boxes


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _subtract_mean(image, rgb_mean):
    image = image.astype(np.float32)
    image -= rgb_mean
    return image / 255.0


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randrange(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if random.randrange(2):
            swap = self.perms[random.randrange(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if random.randrange(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image


class SwapChannels(object):
    '''Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    '''

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        '''
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        '''
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class preproc(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means
        self.rh = RandomHue()
        self.rln = RandomLightingNoise()
        self.rb = RandomBrightness()
        self.rb_crop = RandomBaiduCrop(size=300)
        self.r_crop = RandomCrop(image_size=300)

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"
        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()

        # image = _distort(image)
        image_t = self.rb(image)

        image_hsv = cv2.cvtColor(image_t, cv2.COLOR_BGR2HSV)
        image_hsv = self.rh(image_hsv)
        image_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        image_t = image_bgr

        image_t = self.rln(image_t)

        # image_t = _pad_to_square(image_t, self.rgb_means, True)
        image_t, boxes_t, labels_t = self.r_crop(image_t, boxes, labels)
        image_t, boxes_t = _mirror(image_t, boxes_t)
        image_t = _subtract_mean(image_t, self.rgb_means)

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))
        return image_t, targets_t



