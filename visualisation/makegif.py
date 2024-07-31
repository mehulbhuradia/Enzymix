from PIL import Image, ImageSequence
import os

# List of image filenames
# image_files = os.listdir('./traj/img')
# for i in range(len(image_files)):
#     image_files[i] = './traj/img/'+image_files[i]

# image_files
# print(image_files)

image_files=[]
for i in range(100):
    image_files.append("traj/img/"+str(i)+".png")
image_files.reverse()
print(image_files)

import imageio
import numpy as np

def create_gif(image_list, output_path, duration=1):
    images = []

    for img_path in image_list:
        img = imageio.imread(img_path)
        # Add a blank frame (you can customize the blank frame)
        blank_frame = np.ones_like(img)
        images.append(blank_frame)
        images.append(img)
        
    # Save the gif
    imageio.mimsave(output_path, images, duration=duration)

output_path = "output.gif"
create_gif(image_files, output_path)