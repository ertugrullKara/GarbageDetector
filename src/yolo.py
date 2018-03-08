import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Reshape, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

# Custom Libraries
from . import utils
from .batch_generator import BatchGenerator
from .box import BoundBox, box_iou


class YOLO:
    def __init__(self, config_content, debug=True, weights_found=False):
        """
        Constructor for YOLO network.

        :param config_content: Content of config file.
        :type config_content: Dictionary
        :param debug: Open the debug or close it.
        :type debug: Boolean
        :param weights_found: Are there pre-trained weights.
        :type weights_found: Boolean
        """
        self._debug = debug
        self._custom_weights = weights_found
        self.connections = config_content["connections"][0]
        self.cfg = config_content["net"]
        self.labels = utils.label_names_as_list()
        self.classes = len(self.labels)  # Number of classes
        self.class_wt = np.ones(self.classes, dtype='float32')
        self.max_box_per_image = 3

        self.width = self.cfg["width"]
        self.height = self.cfg["height"]
        self.batch_size = self.cfg["batch"]
        self.decay = self.cfg["decay"]
        self.learning_rate = self.cfg["learning_rate"]
        self.nb_epoch = self.cfg["nb_epoch"]
        self.train_times = self.cfg["train_times"]
        self.valid_times = self.cfg["valid_times"]

        self.threshold = self.connections["region"]["threshold"]
        self.anchors = self.connections["region"]["anchors"]
        self.jitter = self.connections["region"]["jitter"]  # Jitter while generating batches.
        self.num_boxes = self.connections["region"]["num"]  # Number of box guesses for each grid.

        self.__construct_network__(self.cfg["connection_orders"], self.connections)

    def __construct_network__(self, orders, connections):
        """
        Constructs the network according to config file.

        :param orders: Order of the connections.
        :type orders: List
        :param connections: Connections of the network.
        :type connections: Dictionary
        """
        x = input_image = Input(shape=(self.width, self.height, 3))
        self.true_boxes = Input(shape=(1, 1, 1, self.max_box_per_image, 4))
        for order in orders:
            conn = connections[order]
            if "convolution" in order:
                x = Conv2D(conn["filters"], conn["size"], strides=conn["stride"]
                           , padding='same', use_bias=False)(x)

                if conn["batch_normalize"]:
                    x = BatchNormalization()(x)

                if conn["activation"] == "leaky":
                    x = LeakyReLU(alpha=0.1)(x)

                elif conn["activation"] == "linear":
                    pass

            if "maxpool" in order:
                x = MaxPooling2D(pool_size=conn["size"], strides=conn["stride"], padding='same')(x)

            if "region" in order:
                # Last layer. Region layer's properties are implemented in loss function.
                # Instead define 'last(object detection)' layer
                # Object detection layer -> Conv - Reshape - Lambda

                self.feature_extractor = Model(input_image, x)
                if not self._custom_weights:
                    self.feature_extractor.load_weights(self.cfg["feature_extractor_weights"])
                features = self.feature_extractor(input_image)
                self.grid_h, self.grid_w = self.feature_extractor.get_output_shape_at(-1)[1:3]

                filter_num = (conn["coords"] + 1 + self.classes) * conn["num"]
                x = Conv2D(filter_num, 1, strides=1, padding="same", kernel_initializer="lecun_normal")(features)
                x = Reshape((self.grid_h, self.grid_w, conn["num"], conn["coords"] + 1 + self.classes))(x)
                x = Lambda(lambda args: args[0])([x, self.true_boxes])

        self.model = Model([input_image, self.true_boxes], x)

        # Initialize trainable params.
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        # for lay in self.model.layers[:-4]:
        #     lay.trainable = False

        new_kernel = np.random.normal(size=weights[0].shape) / (self.grid_h * self.grid_w)
        new_bias = np.random.normal(size=weights[1].shape) / (self.grid_h * self.grid_w)

        layer.set_weights([new_kernel, new_bias])

        # self.model.summary()

    def load_weights(self, weight_path):
        """

        :param weight_path: Weight path.
        :type weight_path: String
        """
        self.model.load_weights(weight_path)

    def yolo_loss(self, y_true, y_pred):
        """
        Loss function or 'Region Layer' for YOLO.

        :param y_true: True values.
        :param y_pred: Predicted values.
        :return: Loss value.
        :rtype: Float
        """
        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(
            tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, self.num_boxes, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)

        total_recall = tf.Variable(0.)

        """
        Adjust prediction
        """
        ### adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.num_boxes, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.to_int32(y_true[..., 5])

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale

        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale / 2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_bs),
                                                       lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                                true_box_wh + tf.ones_like(true_box_wh) * np.reshape(
                                                                    self.anchors,
                                                                    [1, 1, 1, self.num_boxes, 2]) * no_boxes_mask,
                                                                tf.ones_like(coord_mask)],
                                                       lambda: [true_box_xy,
                                                                true_box_wh,
                                                                coord_mask])

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = loss_xy + loss_wh + loss_conf + loss_class

        if self._debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > self.threshold))

            current_recall = nb_pred_box / (nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
            loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

        return loss

    def train(self, train_imgs, validation_imgs, model_savename="sa.h5"):
        """
        Start training the model.

        :param train_imgs: Traing images in the form of dictionary objects. -Refer to utils.py-
        :type train_imgs: List
        :param validation_imgs: Validation images in the form of dictionary objects. -Refer to utils.py-
        :type validation_imgs: List
        :param model_savename: Model name to be saved.
        :type model_savename: String
        """
        self.coord_scale = self.connections["region"]["coord_scale"]
        self.object_scale = self.connections["region"]["object_scale"]
        self.no_object_scale = self.connections["region"]["noobject_scale"]
        self.class_scale = self.connections["region"]["class_scale"]
        self.warmup_bs = self.connections["region"]["warmup_bs"]

        # Model compilation
        optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=self.decay)
        self.model.compile(loss=self.yolo_loss, optimizer=optimizer)

        # Make dataset generator for train and validation sets.
        generator_cfg = {
            'IMAGE_H': self.height,
            'IMAGE_W': self.width,
            'GRID_H': self.grid_h,
            'GRID_W': self.grid_w,
            'BOX': self.num_boxes,
            'LABELS': self.labels,
            'CLASS': self.classes,
            'ANCHORS': self.anchors,
            'BATCH_SIZE': self.batch_size,
            'TRUE_BOX_BUFFER': self.max_box_per_image,
            "JITTER": self.jitter
        }
        train_batch = BatchGenerator(train_imgs,
                                     generator_cfg,
                                     norm=utils.image_normalizer)
        valid_batch = BatchGenerator(validation_imgs,
                                     generator_cfg,
                                     norm=utils.image_normalizer)

        # Set callbacks
        early_stop = EarlyStopping(monitor='val_loss',
                                   # min_delta=0.001,
                                   patience=10,
                                   mode='auto',
                                   verbose=0)

        checkpoint = ModelCheckpoint(model_savename,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     period=1)

        tb_counter = len([log for log in os.listdir(os.path.expanduser('logs/')) if 'yolo' in log]) + 1
        tensorboard = TensorBoard(log_dir=os.path.expanduser('logs/') + 'yolo' + '_' + str(tb_counter),
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=False)

        # Start the training process
        self.model.fit_generator(generator=train_batch,
                                 steps_per_epoch=len(train_imgs) / self.batch_size,
                                 epochs=self.nb_epoch,
                                 verbose=1,
                                 validation_data=valid_batch,
                                 validation_steps=len(valid_batch) / self.batch_size,
                                 callbacks=[checkpoint, tensorboard],
                                 workers=12,
                                 max_queue_size=8)

    def __sigmoid__(self, x):
        return 1. / (1. + np.exp(-x))

    def __softmax__(self, x, axis=-1, t=-100.):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x / np.min(x) * t

        e_x = np.exp(x)

        return e_x / e_x.sum(axis, keepdims=True)

    def __read_network_output__(self):
        """
        Read the network output and create [BoundBox, ...] list from the found boxes.
        :return: [BoundBox, ...]
        :rtype: List
        """
        grid_h, grid_w, nb_box = self.netout.shape[:3]

        boxes = []

        # decode the output by the network
        self.netout[..., 4] = self.__sigmoid__(self.netout[..., 4])
        self.netout[..., 5:] = self.netout[..., 4][..., np.newaxis] * self.__softmax__(self.netout[..., 5:])
        self.netout[..., 5:] *= self.netout[..., 5:] > self.threshold

        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(nb_box):
                    # from 4th element onwards are confidence and class classes
                    classes = self.netout[row, col, b, 5:]

                    if np.sum(classes) > 0:
                        # first 4 elements are x, y, w, and h
                        x, y, w, h = self.netout[row, col, b, :4]

                        x = (col + self.__sigmoid__(x)) / grid_w  # center position, unit: image width
                        y = (row + self.__sigmoid__(y)) / grid_h  # center position, unit: image height
                        w = self.anchors[2 * b + 0] * np.exp(w) / grid_w  # unit: image width
                        h = self.anchors[2 * b + 1] * np.exp(h) / grid_h  # unit: image height
                        confidence = self.netout[row, col, b, 4]

                        box = BoundBox(confidence, classes, [x, y, w, h])

                        boxes.append(box)

        # suppress non-maximal boxes
        for c in range(self.classes):
            sorted_indices = list(reversed(np.argsort([box.class_probs[c] for box in boxes])))

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]

                if boxes[index_i].class_probs[c] == 0:
                    continue
                else:
                    for j in range(i + 1, len(sorted_indices)):
                        index_j = sorted_indices[j]

                        if box_iou(boxes[index_i], boxes[index_j]) >= self.threshold:
                            boxes[index_j].class_probs[c] = 0

        # remove the boxes which are less likely than a obj_threshold
        boxes = [box for box in boxes if box.get_score() > 0.5]

        return boxes

    def predict(self, image):
        """
        Predict bounding boxes from the given image.

        :param image: Image to predict.
        :type image: cv2.Image
        :return: [BoundBox, ...]
        :rtype: List
        """
        image = cv2.resize(image, (self.width, self.height))
        image = utils.image_normalizer(image)

        input_image = image[:, :, ::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1, 1, 1, 1, self.max_box_per_image, 4))

        self.netout = self.model.predict([input_image, dummy_array])[0]
        boxes = self.__read_network_output__()

        # input_image = image[:, :, ::-1]
        return boxes
