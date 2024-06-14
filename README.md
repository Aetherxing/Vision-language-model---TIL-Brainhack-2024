# Brainhack TIL-AI 2024 Beginner Track VLM

This is a vision language model. When user inputs an image and a caption describing a target object in that image based on the colours of that image, a bounding box corresponding to the target's location within the image will be returned. 

Full ranges of colors of:
- black
- brown
- blue
- gray
- green
- orange
- pink
- purple
- red
- white
- yellow

is able to be detected. 

Model able to detect objects with descriptions of up to 2 different colors. 

Originally intended for model to be able to uniquely identify objects of 3 classes of aircrafts: 
1. Drones
2. Airplanes
3. Helicopters

--> Function failed as training model was unable to recognise class of object. 

# Purpose of each file

aircraft_classifier.py is training model to identify class of aircraft. 
color_classifier.py is training model to identify color of target object. 
aircraft_classifier_finale2.h5 & color_classifier_finale.h5 are final versions of trained models. 

locator.py is an outdated version of the target in image locator code, please ignore it and refer to locator2.py
