### This is cyan mask version ###

"""
Mask R-CNN
Train on the toy Custom dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 custom.py train --dataset=/path/to/custom/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 custom.py train --dataset=/path/to/custom/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 custom.py train --dataset=/path/to/custom/dataset --weights=imagenet

    # Apply color splash to an image
    python3 custom.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 custom.py splash --weights=last --video=<URL or path to file>
"""

from fileinput import filename
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw


#############################################################
# try to fix random result
############################################################
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import keras
import tensorflow as tf
# config = tf.ConfigProto()

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 0.6 sometimes works better for folks
keras.backend.set_session(tf.compat.v1.Session(config=config))

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "hands_dolls"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + custom

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.95


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the Custom dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("custom", 1, "doll")
        self.add_class("custom", 2, "hand")
        # self.add_class("objects", 2, "hand")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # coco annotations saves images and their infos in the form:
        # { "info": {...},
        #   "licenses": [{...}],
        #   "images": [
        #       {
        #           "file_name": "JPEGImages/image.jpg"
        #           ...
        #       },  ...
        #   ],
        #   "annotations": [
        #       {
        #           "id": 0,
        #           "image_id": 0,
        #           "category_id": 2,
        #           "segmentation": [
        #               [
        #                   x0,
        #                   y0,
        #                   x1,
        #                   y1,
        #                   ...
        #               ]          
        #   ...
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        # annotations = list(annotations.values())  # don't need the dict keys
        annot = annotations["annotations"]
        # get filename and split JPEGImages/image.jpg to image.jpg
        filenames = [imgs_info["file_name"] for imgs_info in annotations["images"]]
        filenames = [fn.split("/")[-1] for fn in filenames]
        
        # Add images
        # for a in annotations:
        #     # Get the x, y coordinaets of points of the polygons that make up
        #     # the outline of each object instance. These are stores in the
        #     # shape_attributes (see json format above)
        #     # The if condition is needed to support VIA versions 1.x and 2.x.
        #     if type(a['regions']) is dict:
        #         polygons = [r['shape_attributes'] for r in a['regions'].values()]
        #     else:
        #         polygons = [r['shape_attributes'] for r in a['regions']] 

        #     # load_mask() needs the image size to convert polygons to masks.
        #     # Unfortunately, VIA doesn't include it in JSON, so we must read
        #     # the image. This is only managable since the dataset is tiny.
        #     image_path = os.path.join(dataset_dir, a['filename'])
        #     image = skimage.io.imread(image_path)
        #     height, width = image.shape[:2]
        
        # make a dict to map annot_id to their image_id
        img_to_annot = {}
        for a in annot:
            img_id = a["image_id"]
            if img_id in img_to_annot:
                img_to_annot[img_id].append(a["id"])
            else:
                img_to_annot[img_id] = [a["id"]]
        
        image_ids = list(img_to_annot.keys())
        for img_id in image_ids:
            new_annot = []
            num_ids = []
            for a_id in img_to_annot[img_id]:
                try:
                    num_ids.append(annot[a_id]["category_id"])
                except:
                    pass

                points = [int(item) for item in annot[a_id]["segmentation"][0]]
                points_x = points[::2]
                points_y = points[1::2]
                new_dict = {'name': 'polygon', 'all_points_x': points_x, 'all_points_y': points_y}
                new_annot.append(new_dict)

            image_path = os.path.join(dataset_dir, filenames[img_id])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            
            # fill up image_info
            self.add_image(
                "custom",
                image_id=filenames[img_id],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=new_annot,
                num_ids=num_ids)
                
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a custom dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "custom":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = image_info['num_ids']

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        num_ids = np.array(num_ids, dtype=np.int32)
        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "custom":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')


# def color_splash(image, mask):
#     """Apply color splash effect.
#     image: RGB image [height, width, 3]
#     mask: instance segmentation mask [height, width, instance count]

#     Returns result image.
#     """
#     # Make a grayscale copy of the image. The grayscale copy still
#     # has 3 RGB channels, though.
#     gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
#     # Copy color pixels from the original color image where mask is set
#     if mask.shape[-1] > 0:
#         # We're treating all instances as one, so collapse the mask into one layer
#         mask = (np.sum(mask, -1, keepdims=True) >= 1)
#         splash = np.where(mask, image, gray).astype(np.uint8)
#     else:
#         splash = gray.astype(np.uint8)
#     return splash

def color_splash(image, mask):
    import cv2
    # Apply mask effect
    height, width = image.shape[:2]
    cyan = np.full_like(image,(0,255,255)).astype(np.uint8)
    # color[:,:,1] = 255 # Green in BGR format

    if mask.shape[-1] > 0:
        blend = 0.5
        img_cyan = cv2.addWeighted(image, blend, cyan, 1-blend, 0)
        # We treat all instance as one, so let the mask be one layer
        # if we need to identify two hands then need more layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, img_cyan, image).astype(np.uint8)
    else:
        splash = image.astype(np.uint8)
    return splash
    # """Apply color splash effect.
    # image: RGB image [height, width, 3]
    # mask: instance segmentation mask [height, width, instance count]
    # Returns result image.
    # """
    # # Make a grayscale copy of the image. The grayscale copy still
    # # has 3 RGB channels, though.
    # gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # # Copy color pixels from the original color image where mask is set
    # if mask.shape[-1] > 0:
    #     # We're treating all instances as one, so collapse the mask into one layer
    #     mask = (np.sum(mask, -1, keepdims=True) >= 1)
    #     splash = np.where(mask, image, gray).astype(np.uint8)
    # else:
    #     splash = gray.astype(np.uint8)
    # return splash

def detect_and_color_splash(model, image_path=None, video_path=None, stream_path=None):
    assert image_path or video_path or stream_path

    # Image or video or stream?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        # extract filename
        file_name = image_path.split(".")[-2]
        if len(file_name.split("/")) > 1:
            file_name = file_name.split("/")[-1]
        file_name = 'output_{}.png'.format(file_name)
        skimage.io.imsave(file_name, splash)
        print("Saved to ", file_name)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        # file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        file_name = video_path.split(".")[-2]
        if len(file_name.split("/")) > 1:
            file_name = file_name.split("/")[-1]
        file_name = 'output_{}.avi'.format(file_name)
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
        print("Saved to ", file_name)
    elif stream_path:
        # print("stream path:{}".format(stream_path))
        # stream_path = stream_path.split('\"')[1]
        import cv2
        from threading import Thread
        import tensorflow as tf
        print('######## stream path:'+ stream_path)
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        
        class VideoStreamWidget(object):
            def __init__(self, src=0):
                # Create a VideoCapture object
                self.capture = cv2.VideoCapture(src)

                # Start the thread to read frames from the video stream
                self.thread = Thread(target=self.update, args=())
                self.thread.daemon = True
                self.thread.start()

            def update(self):
                # Read the next frame from the stream in a different thread
                while True:
                    if self.capture.isOpened():
                        (self.status, self.frame) = self.capture.read()

            def show_frame(self):
                # Display frames in main program
                if self.status:
                    # OpenCV returns images as BGR, convert to RGB
                    image = self.frame[..., ::-1]
                    # Detect objects
                    r = model.detect([image], verbose=0)[0]
                    # # Color splash
                    splash = color_splash(image, r['masks'])
                    # # RGB -> BGR to save image to video
                    self.frame = splash[..., ::-1]
                    # self.frame = self.maintain_aspect_ratio_resize(self.frame, width=600)
                    cv2.imshow('IP Camera Video Streaming', self.frame)

        video_stream_widget = VideoStreamWidget(stream_path)
        count = 0
        while True:
            print("frame: ", count)
            count += 1
            try:
                video_stream_widget.show_frame()
                # Press Q on keyboard to stop recording
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    video_stream_widget.capture.release()
                    cv2.destroyAllWindows()
                    break
            except AttributeError:
                pass

        
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the Custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--rtsp', required=False,
                        metavar="path or URL to rtsp",
                        help='rtsp to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video or args.rtsp,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        
        model.load_weights(weights_path, by_name=True)
        

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video, stream_path=args.rtsp)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
