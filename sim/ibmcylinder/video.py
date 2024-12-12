
import cv2
import os
import glob

# Directory containing the images
image_folder = "/home/spencer/ml-airfoil/sim/ibmcylinder/pmovie"  # Replace with the path to your folder
output_video = "output_video.mp4"

# Get all image files sorted by their numeric order
images = sorted(glob.glob(os.path.join(image_folder, "*.png")),
                key=lambda x: float(os.path.splitext(os.path.basename(x))[0].split('-')[-1]))

# Get the width and height of the first image
frame = cv2.imread(images[0])
height, width, layers = frame.shape
fps = 24  # Frames per second (adjust as needed)

# Create the VideoWriter object
video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for image in images:
    img = cv2.imread(image)
    video.write(img)

video.release()
print(f"Video created successfully: {output_video}")
