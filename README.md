# Semantic Segmentation on Road Scenes

1. Clone this repository or download zip file.
2. Extract the contents of the zip
3. Open Anaconda Command Prompt and navigate to repository on local disk using cd
4. pip install imutils  (IMPORTANT)

# To apply model on screen capture:

Use the command: "python segment_screen.py" without quotes

The mss library has been used for screen capture

Modify line 60 of segment_screen.py to adjust resolution of screen capture

Line 60 : monitor = {"top": 0, "left": 0, "width": 640, "height": 480}
This line means that the top-left part of the screen will be captured whose dimensions are 640x480

Tip: It is recommended to place screen capture window on the bottom right of the screen.

# To apply model on image:

Use the command: "python segment_image.py --image images/example_01.png" without quotes

Some example images have been included in this repository

For custome images, place the image in the ./images folder and change the parameter of the command accordingly

# To apply model on video:

Use the command: "python segment_video.py --video videos/massachusetts.mp4 --output output massachusetts_output.avi" without quotes

The video file massachusetts.mp4 has been included in this repository.

For custom videos place videos in ./videos. Also give an appropriate name to the output file where the video file will be written to disk with the model applied on every frame.