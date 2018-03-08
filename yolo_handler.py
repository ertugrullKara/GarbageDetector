# import skvideo.io as cv2
import argparse
import os

import cv2

from src import image_util
from src import utils
from src import yolo


class YoloHandler:
    def __init__(self, argparser=None):
        """
        Constructor for YoloHandler.
        Takes argparser object. If no args provided, continue with the default setup.

        :param argparser: arguments
        :type argparser: argparse.ArgumentParser
        """
        if argparser:
            self.debug = argparser.debug
            self.action = argparser.action
            self.input_type = argparser.type
            self.input_path = argparser.name
            self.config_path = argparser.config
            self.proccess_image = True  # Show image on screen.
            self.__init_check__()
        else:
            self.debug = True
            self.action = None
            self.input_type = None
            self.input_path = None
            self.config_path = None
            self.proccess_image = False  # Do not show image on screen.

        self.__clear__()

        self.__load_cfg__()
        self.__load_model__()

    def __clear__(self):
        """
        Clear instance variables that are used for predictions.
        """
        self._image_original_width = 416
        self._image_original_height = 416
        self.image = None
        self.boxes = []

    def __init_check__(self):
        """
        Check arguments that are provided to prevent passing erroneous data.
        """
        if not self.action:
            print("Invalid action type!")
            exit(1)
        if self.action.lower() == "train":
            pass
        elif self.action.lower() == "validate":
            if not self.input_type or not self.input_path:
                print("Please provide input name and type")
                exit(1)
        else:
            print("Invalid action type!")
            exit(1)

    def __load_cfg__(self):
        """
        Load the yolo config file.
        """
        if self.config_path:
            self.config = utils.config_file_as_dict(self.config_path)
        else:
            self.config = utils.config_file_as_dict()
        self.model_name = self.config["net"]["best_model_weights"]

    def __load_model__(self):
        """
        Load the model. Load from pre-trained model if exists.
        """
        self.saved_weights_found = False
        if os.path.exists(self.model_name):
            self.saved_weights_found = True

        # Construct the model.
        self.model = yolo.YOLO(self.config, debug=self.debug, weights_found=self.saved_weights_found)
        # Load the pre-trained weights (if any).
        if self.saved_weights_found:
            print("Loading pre-trained weights from", self.model_name)
            self.model.load_weights(self.model_name)

    def __train_action__(self):
        """
        Imports validation and train datasets. Then starts to train the model.
        """
        # Import images to train and validate the network..
        valid_annots = utils.images_with_annots("validate")
        train_annots = utils.images_with_annots("traing")
        # Start training.
        self.model.train(train_annots, valid_annots, self.model_name)

    def __predict_image__(self):
        """
        Predicts image from the model. Saves the found boxes in self.boxes.
        """
        self.boxes = self.model.predict(self.image)

    def __show_prediction__(self):
        """
        Show prediction result.
        """
        image_util.show_image(self.image, self.boxes,
                              self._image_original_width,
                              self._image_original_height)
        cv2.waitKey(4)  # Approx. 24FPS

    def __read_video__(self):
        """
        Open the specified input file path. Path is either file or URL according to cv2.
        Read the image, make predictions and show them on the screen. Also outputs to console.
        Used for testing purposes.

        :return: Successful or not.
        :rtype: Boolean
        """
        vidcap = cv2.VideoCapture(self.input_path)
        if not vidcap.isOpened():
            return False
        self._image_original_width = int(min(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH), 1024))
        self._image_original_height = int(min(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT), 768))
        print(self._image_original_width, self._image_original_height)
        success, self.image = vidcap.read()
        count = 0
        while success:
            self.__predict_image__()
            self.__show_prediction__()
            if success:
                if len(self.boxes) > 0:
                    print("Found {} boxes".format(len(self.boxes)))
            count += 1
            success, self.image = vidcap.read()
        vidcap.release()
        cv2.destroyAllWindows()
        return True

    def __test_on_image__(self):
        """
        Used for testing on a single image.
        """
        if not self.image:
            self.image = cv2.imread(self.input_path)
        self.__predict_image__()
        if self.proccess_image:
            self.__show_prediction__()
        cv2.waitKey(0)

    def __test_action__(self):
        """
        Decides next step to take by checking the input type provided by argparser.
        """
        if self.input_type == "image":
            self.__test_on_image__()
        elif self.input_type == "video":
            self.__read_video__()
        elif self.input_type == "set":
            if self.action.lower() == "validate":
                _imgs = utils.images_with_annots("validate")
            else:
                _imgs = utils.images_with_annots("test")
            for img in _imgs:
                self.image = img
                self.__test_on_image__()
        else:
            print("Invalid action type! Abort!")

    def _execute_action(self):
        """
        Decides next step to take by checking the action type provided by argparser.
        """
        if self.action.lower() == "train":
            self.__train_action__()
        elif self.action.lower() == "validate":
            self.__test_action__()
        elif self.action.lower() == "test":
            self.__test_action__()
        exit(1)

    def send_predict_results_from_path(self, type, input_path, defining_stamp):
        """
        API like function. Send in parameters and get back result. Not a single image actually shown.
        To use/test from here.

        :param type: "image" or "video" as string
        :param input_path: path to image or video.
        :param defining_stamp: defining property of image to keep track of boxes returned in realtime processing.
            ie. timestamp
        :return: provided defining_stamp reflected. Also list of found boxes in the form of BoundBox objects.
        :rtype: Tuple(defining_stamp, [BoundBox, ...])
        """
        self.__clear__()
        self.input_type = type
        self.input_path = input_path
        self.__test_action__()
        return defining_stamp, self.boxes

    def send_predict_results_from_image(self, type, image, defining_stamp):
        """
        API like function. Send in parameters and get back result. Not a single image actually shown.
        To use from outside.

        :param type: "image" as string
        :param image: actual image array. Output of cv2.imread(IMAGE_PATH)
        :param defining_stamp: defining property of image to keep track of boxes returned in realtime processing.
            ie. timestamp
        :return: provided defining_stamp reflected. Also list of found boxes in the form of BoundBox objects.
        :rtype: Tuple(defining_stamp, [BoundBox, ...])
        """
        self.__clear__()
        self.input_type = type
        self.image = image
        self.__test_action__()
        return defining_stamp, self.boxes


def _main(args):
    yolo_handler = YoloHandler(args)
    yolo_handler._execute_action()


def __str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='Train and validate yolo model.')

    argparser.add_argument(
        '-a',
        '--action',
        help='"train" on default dataset or "validate" on given input.')

    argparser.add_argument(
        '-t',
        '--type',
        help='input type: "image" or "video"',
        default="set"
    )

    argparser.add_argument(
        '-n',
        '--name',
        help='input name: image name or video name/url')

    argparser.add_argument(
        '-c',
        '--config',
        help='path to the configurations file.')

    argparser.add_argument(
        '-d',
        '--debug',
        type=__str2bool,
        nargs='?',
        const=True,
        help='Open the debug output or not. Boolean value.',
        default=False
    )

    _main(argparser.parse_args())
