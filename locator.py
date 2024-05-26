import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define RGB ranges for each color category
color_ranges = {
    'black': ([0, 0, 0], [50, 50, 50]),
    'brown': ([101, 67, 33], [165, 105, 60]),
    'blue': ([0, 0, 150], [50, 50, 255]),
    'gray': ([100, 100, 100], [180, 180, 180]),
    'green': ([0, 100, 0], [50, 180, 50]),
    'orange': ([200, 100, 0], [255, 165, 0]),
    'pink': ([200, 100, 150], [255, 192, 203]),
    'purple': ([100, 0, 100], [180, 50, 180]),
    'red': ([150, 0, 0], [255, 50, 50]),
    'white': ([200, 200, 200], [255, 255, 255]),
    'yellow': ([200, 200, 0], [255, 255, 100])
}

def identify_color_objects(image, color_names):
    # Initialize empty list to store individual bounding boxes
    bounding_boxes = []
    
    # Iterate over each specified color
    for color_name in color_names:
        # Get the color range for the current color
        lower_range, upper_range = color_ranges[color_name]
        
        # Create a mask for the current color range
        mask = cv2.inRange(image, np.array(lower_range), np.array(upper_range))
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes around the contours
        for contour in contours:
            # Get the bounding box for the current contour
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
    
    # If no objects are detected, return the original image
    if not bounding_boxes:
        return image, None
    
    # Calculate the overall bounding box that encompasses all detected objects
    x_min = min(box[0] for box in bounding_boxes)
    y_min = min(box[1] for box in bounding_boxes)
    x_max = max(box[0] + box[2] for box in bounding_boxes)
    y_max = max(box[1] + box[3] for box in bounding_boxes)
    
    # Draw the overall bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    
    return image, (x_min, y_min, x_max, y_max)

def read_color_names_from_file(file_path):
    with open(file_path, 'r') as file:
        color_names = file.read().strip().split(',')
    return color_names

def identify_color_objects_in_image(image_path, color_names_file):
    # Read color names from the text file
    color_names = read_color_names_from_file(color_names_file)
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert image to RGB (OpenCV reads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Identify objects of the specified colors and draw one bounding box around them
    result_image, overall_bbox = identify_color_objects(image_rgb, color_names)
    
    # Display the result
    plt.imshow(result_image)
    plt.axis('off')
    
    # Draw the overall bounding box if objects are detected
    if overall_bbox:
        x_min, y_min, x_max, y_max = overall_bbox
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                           linewidth=2, edgecolor='r', facecolor='none'))
    
    plt.show()

# Manually input the paths to the image and the text file containing color names
image_path = input("Enter the path to the image file: ")
color_names_file = input("Enter the path to the text file containing color names: ")

# Call the function to identify color objects in the image
identify_color_objects_in_image(image_path, color_names_file)