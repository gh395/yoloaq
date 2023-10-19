# yoloaq
YOLO for Emission Source Detection

Object detection for satellite images of a world coordinate using Google Maps

Labels: paved parking lot, large parking lot, unpaved parking lot, unpaved area with trucks, unpaved area, facility with trucks, airport facilities

dload_dataset_pipeline.py is used to run the specified models on given world coordinate and the results will be placed in the outputs folder.

If you would like to use this script, you must generate your own api key from Google's Maps Static API and place it in a .env file.

Run using the command lines with the following flags, (* indicates required flags):
```
-lat (latitude)*
-lng (longitude)*
-n (num rows)*
-m (num cols)*
-z (zoom level, defaults to 18 default)
-rm (run multilabel model)
-rb (run binary model)
-ry (run yolo model)
-conf (confidence threshold for YOLO model, defaults to 0.1))
-outm (output folder name for multilabel model, defaults to name of model file)
-outb (output folder name for binary model, defaults to name of model file)
```

Example usage:
```
python dload_dataset_pipeline.py -lat 51.525260 -lng 0.127625 -n 3 -m 5 -z 16 -rm multi.h5 -rb unpaved_area.h5 -ry yolo.pt -conf 0.05
```

Most of the binary model files were not included due to git lfs storage restrictions

Example outputs are provided in the outputs folder
![YOLOexample](https://github.com/gh395/yoloaq/blob/main/outputs/49.633854_6.177700_2by2/yolo/merged_indiv.png?raw=true)
