import copy

import cv2
import numpy as np
from imgaug import augmenters as iaa
from tensorflow.python.keras.utils import Sequence

from .box import BoundBox, box_iou


class BatchGenerator(Sequence):
    def __init__(self, images,
                 config,
                 shuffle=True,
                 norm=None):
        """
        :param images: Images in the form of dictionary objects. -Refer to utils.py-
        :type images: List
        :param config: Config for the generator. -Refer to yolo_handler.py-
        :type config: Dictionary
        :param shuffle: Enable shuffling.
        :type shuffle: Boolean
        :param norm: Normalizer function, if provided..
        :type norm: Function
        """
        self.generator = None

        self.images = images
        self.config = config
        self.shuffle = shuffle
        self.jitter = self.config["JITTER"]
        self.norm = norm

        self.counter = 0
        # Create boxes for anchor values to find best fitting anchor box for true and predicting boxes.
        self.anchors = [BoundBox(None, None, [0, 0,
                                              self.config["ANCHORS"][2 * i],
                                              self.config["ANCHORS"][2 * i + 1]]) for i in
                        range(int(len(self.config["ANCHORS"]) / 2))]

        ### augmentors by https://github.com/aleju/imgaug

        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                # iaa.Fliplr(0.5), # horizontally flip 50% of all images
                # iaa.Flipud(0.2), # vertically flip 20% of all images
                # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    # rotate=(-5, 5), # rotate by -45 to +45 degrees
                    # shear=(-5, 5), # shear by -16 to +16 degrees
                    # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                               # search either for all edges or for directed edges
                               # sometimes(iaa.OneOf([
                               #    iaa.EdgeDetect(alpha=(0, 0.7)),
                               #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                               # ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               # iaa.Invert(0.05, per_channel=True), # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               # change brightness of images (50-150% of original value)
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               # iaa.Grayscale(alpha=(0.0, 1.0)),
                               # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                               # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / self.config['BATCH_SIZE']))

    def __getitem__(self, idx):
        """
        Get item for batch.

        :param idx: Index to fetch.
        :type idx: Integer
        :return: The next batch.
        x_batch: input image to
        b_batch: true box
        y_batch: ground truth x, y, w, h, confidence and class probs
        :rtype: [x_batch, b_batch], y_batch
        """
        l_bound = idx * self.config['BATCH_SIZE']
        r_bound = (idx + 1) * self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))  # input images
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config['TRUE_BOX_BUFFER'],
                            4))  # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'], self.config['GRID_W'], self.config['BOX'],
                            4 + 1 + 1))  # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)

            # construct output from object's x, y, w, h
            true_box_index = 0

            for obj in all_objs:
                if obj['name'] in self.config['LABELS']:
                    obj_indx = self.config['LABELS'].index(obj['name'])

                    center_x = (obj["xratio"] * self.config['GRID_W'])
                    center_y = (obj["yratio"] * self.config['GRID_H'])
                    center_w = (obj["wratio"] * self.config['GRID_W'])
                    center_h = (obj["hratio"] * self.config['GRID_H'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x >= self.config['GRID_W'] or grid_y >= self.config['GRID_H']:
                        continue

                    box = [center_x, center_y, center_w, center_h]

                    # find the anchor that best predicts this box
                    best_anchor = -1
                    max_iou = -1

                    shifted_box = BoundBox(None, None, [0, 0, center_w, center_h])

                    for i in range(len(self.anchors)):
                        anchor = self.anchors[i]
                        iou = box_iou(shifted_box, anchor)

                        if max_iou < iou:
                            best_anchor = i
                            max_iou = iou

                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                    y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.
                    y_batch[instance_count, grid_y, grid_x, best_anchor, 5] = obj_indx

                    # assign the true box to b_batch
                    b_batch[instance_count, 0, 0, 0, true_box_index] = box

                    true_box_index += 1
                    true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

            # assign input image to x_batch
            if self.norm != None:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    left = int(self.config['IMAGE_W'] * (obj["xratio"] - obj["wratio"] / 2))
                    top = int(self.config['IMAGE_H'] * (obj["yratio"] - obj["hratio"] / 2))
                    right = int(self.config['IMAGE_W'] * (obj["xratio"] + obj["wratio"] / 2))
                    bottom = int(self.config['IMAGE_H'] * (obj["yratio"] + obj["hratio"] / 2))
                    cv2.rectangle(img[:, :, ::-1], (left, top), (right, bottom),
                                  (255, 0, 0), 3)
                    cv2.putText(img[:, :, ::-1], obj['name'],
                                (left + 2, top + 12),
                                0, 1.2e-3 * img.shape[0],
                                (0, 255, 0), 2)
                cv2.imshow("DEBUG", img)
                cv2.waitKey(0)

                x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1

        self.counter += 1
        # print ' new batch created', self.counter

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)
        self.counter = 0

    def aug_image(self, train_instance, jitter):
        """
        Augments the train instance.

        :param train_instance: Image annotation.
        :type train_instance: Dictionary
        :param jitter: Jitter value.
        :type jitter: Float
        :return: Augmented image and objects in image.
        :rtype: Tuple
        """
        image_name = train_instance['filename']
        image = cv2.imread(image_name)
        h, w, c = image.shape
        flip = 0

        all_objs = copy.deepcopy(train_instance['objects'])

        do_augment = np.random.choice([True, False], p=[jitter, 1.0 - jitter])
        if do_augment:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            ### translate the image
            max_offx = (scale - 1.) * w
            max_offy = (scale - 1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy: (offy + h), offx: (offx + w)]
            ### flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5:
                image = cv2.flip(image, 1)
            image = self.aug_pipe.augment_image(image)

        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:, :, ::-1]

        # fix objects' position and size
        for obj in all_objs:
            left = w * (obj["xratio"] - obj["wratio"] / 2)
            right = w * (obj["xratio"] + obj["wratio"] / 2)
            if do_augment:
                left = int(left * scale - offx)
                right = int(right * scale - offx)
            left = int(left * float(self.config['IMAGE_W']) / w)
            right = int(right * float(self.config['IMAGE_W']) / w)
            left = min(max(int(left), 0), self.config['IMAGE_W'])
            right = min(max(int(right), 0), self.config['IMAGE_W'])
            obj["xratio"] = (float(right + left) / 2) / self.config['IMAGE_W']

            top = h * (obj["yratio"] - obj["hratio"] / 2)
            bottom = h * (obj["yratio"] + obj["hratio"] / 2)
            if do_augment:
                top = int(top * scale - offy)
                bottom = int(bottom * scale - offy)
            top = int(top * float(self.config['IMAGE_H']) / h)
            top = min(max(int(top), 0), self.config['IMAGE_H'])

            bottom = int(bottom * float(self.config['IMAGE_H']) / h)
            bottom = min(max(int(bottom), 0), self.config['IMAGE_H'])
            obj["yratio"] = (float(bottom + top) / 2) / self.config['IMAGE_H']

            if do_augment and flip > 0.5:
                obj["xratio"] = 1 - obj["xratio"]
                pass

        return image, all_objs


if __name__ == "__main__":
    import os
    import utils

    os.chdir(os.path.abspath(".."))

    anchors = [
        1.22,
        1.56,
        2.95,
        2.87,
        4.63,
        5.66,
        7.31,
        8.02,
        10.66,
        8.77
    ]
    generator_cfg = {
        'IMAGE_H': 416,
        'IMAGE_W': 416,
        'GRID_H': 13,
        'GRID_W': 13,
        'BOX': 5,
        'LABELS': utils.label_names_as_list(),
        'CLASS': 1,
        'ANCHORS': anchors,
        'BATCH_SIZE': 4,
        'TRUE_BOX_BUFFER': 3,
        "JITTER": 0
    }
    imgs_annots = utils.images_with_annots()
    train_batch = BatchGenerator(imgs_annots,
                                 generator_cfg)
    for batch in train_batch:
        print(np.nonzero(batch[1]))
        exit(1)
