# TODO: need to make the code more clean
import cv2
import os
image_path = 'img3/'
video_path = 'output3.avi'
fps = 40
frames = []
# loading the files
files = [f for f in os.listdir(image_path) if os.isfile(os.join(image_path, f))]
files.sort(key= lambda x: int(x.split("_")[1].split(".")[0]))

for i in range(len(files)):
    filename = image_path + files[i]
    image = cv2.imread(filename)
    # getting information about the image
    height,width,layers = img.shape
    size = (width,height)
    frames.append(image)
out = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

for i in range(len(frames)):
    out.write(frames[i])
out.release()
