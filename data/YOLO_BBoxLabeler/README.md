# Yolo_mark
**Windows** & **Linux** GUI for marking bounded boxes of objects in images for training Yolo v2
Credit: https://github.com/AlexeyAB/Yolo_mark

* To compile on **Linux** type in console 3 commands:
    ```
    cmake .
    make
    ./linux_mark.sh
    ```

Supported both: OpenCV 2.x and OpenCV 3.x

--------

1. To test, simply run 
  * **on Linux:** `./linux_mark.sh`

2. To use for labeling your custom images:

 * put your `.jpg`-images to the directory ../raw_data/img
 * put names of objects, one for each line in file ../classindex.txt
 * run file: `./linux_mark.cmd`
