Garbage recognition and detection module of GarbageCollector Drone project.




Requirements
-------------
imgaug==0.2.5
tensorflow-gpu==1.4.0
simplejson==3.12.0
numpy==1.13.3
opencv_python==3.3.0.10
Pillow==4.3.0

They can be found in requirements.txt at the root of the project.

And, python3.



Parameters
-------------
`-a --action`: Action to perform. Either 'train' or 'validate'. 
    Used only for testing purposes. Main model will not need this since it will only perform predictions on the video feed.

`-t --type`: Either image or video.

`-n --name`: image path or video path/url.

`-c --config`: if specified, different config location than default (./cfg/tiny_yolo_config.json)

Running
-------------
Run the following to train,

`python3 run_tyolo.py -a train`



Run the followingto train to test on an image or video,

`python3 run_tyolo.py -a validate -t video -n sample.mp4`



Run the following to train to test on validation set,

`python3 run_tyolo.py -a validate`

Run the following to train to test on test set,

`python3 run_tyolo.py -a test`

Config
-------------
Config file can be found in .cfg/ folder inside the `tiny_yolo_config.json` file.
Change the parameters as you wish. But DO NOT forget to set the last convolutional layer's number of filters according to the formula

`(coords + classes + 1) * num`

Data
-------------
Prepare the data and fill the files for training data index and validation data index into the files stated as in `utils.py`
File paths should be relative to the main module path.

For more about preparing the data, refer to 'data' folder.

Upon acquiring new data, it is best to generate new anchors to predict more efficiently.