from flask import Flask, send_file, send_from_directory, flash, request,\
        redirect, render_template, url_for
from app import app
from werkzeug.utils import secure_filename
from object_tracking import extract_images
from object_tracker.sort import Sort
from os.path import join, isfile
from os import getcwd, listdir, makedirs
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import cv2 as cv
from sys import exit
import numpy as np
from random import choices
from colorutils import hex_to_hsv

# allowed file extension
ALLOWED_EXTENSIONS = {'mp4'}

# global variable to hold the filename
filename = ""
# global variable to hold the filename without extension
filename_ext = ""
# global variable to hold the colors of the teams (hsv)
color1 = np.array([0,0,0])
color2 = np.array([0,0,0])
color3 = np.array([0,0,0])
# global variable to hold whether the to terminate the program
cancel = False
# global dictionary that holds the image id and the path to the image
filename_space = {}
current_image_id = 0


def video_to_frame_multiprocessing(filename, chunk_size = 60):
    """Helper function of background_process_task 
    Returns the chunk size
    
    arguments:
        filename -> name of the file
        chunk_size -> size of each video sector
    """
    global filename_ext, color1, color2, color3, cancel
    video_path = join(getcwd(),'static','upload',filename)
    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # reducing the chunk_size if it is greater than
    # the total frame
    # have some better logic here
    if chunk_size > total_frames:
        chunk_size = int(chunk_size / total_frame)

    chunks = [[i,i+chunk_size] for i in range(0,total_frames,chunk_size)]
    # making sure that the last element in the array is total_frames
    chunks[-1][-1] = min(chunks[-1][-1],total_frames)
    print(f"Video is divided into {chunks}\n")
    return video_path, chunks[:1]


def frame_to_video():
    """Combines the Frames to Video
    """
    global filename, filename_ext
    image_path = join(getcwd(),'static','images',filename_ext)
    video_path = join(getcwd(),'static','video',f"{filename_ext}.mp4")
    fps = 40
    frames = []
    files = [f for f in listdir(image_path) if isfile(join(image_path,f))]
    files.sort(key= lambda x: int(x.split("_")[1].split(".")[0]))
    for i in range(len(files)):
        filename = join(image_path, files[i])
        image = cv.imread(filename)
        # getting information about the image
        height,width,layers = image.shape
        size = (width,height)
        frames.append(image)
    out = cv.VideoWriter(video_path,cv.VideoWriter_fourcc(*'MP4V'), fps, size)

    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

def allowed_file(filename):
    """Determines whether 'filename' is an allowed filename

    ! extensions can be added in the global variable ALLOWED_EXTENSIONS
    """
    return "." in filename and \
            filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/imagepreview')
def imagepreview():
    """shows a preview of images while the images are being loaded
    """
    global filename_ext, filename_space, current_image_id
    image_path = join(getcwd(),'static','images',filename_ext)
    for image in listdir(image_path):
        # checking if the 'image' is a file
        if isfile(join(image_path, image)):
            # file_namespace = {0 : path/img_0.jpg, "
            filename_space[int(image.rsplit(".")[0][4:])] = join('static',
                    'images', filename_ext, image)
    return render_template('imagepreview.html', image = filename_space,
            key=current_image_id)


@app.route('/imagepreview/next/<int:key>')
def next(key):
    """Helper function of imagepreview
    Redirects to the next image
    
    arguments
        key: previous image id of the image
    """
    global current_image_id, filename_space
    error = None
    # checking if the current_image is present
    print(key)
    try:
        temp_path = filename_space[current_image_id]
        current_image_id = key + 1
    except KeyError:
        # if key error means image isnt present, show the same image
        # flash a message
        print("error mate")
        error = "file is not present, try again later"
        return render_template('imagepreview.html', image = filename_space,
                key=current_image_id, error=error)
    return redirect(url_for('imagepreview'))

    
@app.route('/imagepreview/prev/<int:key>')
def prev(key):
    """Helper function of imagepreview
    Redirects to the prev image

    arguments
        key: previous image id of the image
    """
    global current_image_id, filename_space
    if key == 0:
        current_image_id = 0
    else:
        current_image_id = key - 1 
    return redirect(url_for('imagepreview'))


@app.route('/finalvideo')
def finalvideo():
    """Contains the images of the final
    Video plus the video itself!
    """
    global filename_ex
    frame_to_video()
    image_path = join(getcwd(),'static','images',filename_ext)
    only_files = [ join('static','images',filename_ext,f) for f in listdir(image_path)\
            if isfile(join(image_path,f))]
    # make clean u after user exits
    return render_template('finalvideo.html',filename = f"{filename_ext}.mp4",images=choices(only_files,k = 4))


@app.route('/loader',methods=['POST','GET'])
def loader():
    """
    """
    return render_template('loader.html', filename=filename)


@app.route('/', methods=['POST', 'GET'])
def verification():
    """
    """
    global filename, color1, color2, color3
    if request.method == 'POST':
        # converting hex to hsv
        color1 = np.array(list(hex_to_hsv(request.form.get('team1'))))
        color2 = np.array(list(hex_to_hsv(request.form.get('team2'))))
        color3 = np.array(list(hex_to_hsv(request.form.get('team3'))))
        # changing the decimal values to percentages
        color1[1] *= 100
        color2[1] *= 100
        color3[1] *= 100
        color1[2] *= 100
        color2[2] *= 100
        color3[2] *= 100
        if 'file' not in request.files:
            flash('no file selected')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            _filename = secure_filename(file.filename)
            file.save(join(app.config['UPLOAD_FOLDER'], _filename))
            filename = _filename
            return redirect(url_for('loader'))
    return render_template('uploadvideo.html')


@app.route('/show/<filename>')
def showVideo(filename):
    """
    """
    global filename_ext
    return redirect(url_for('static',filename =
        join('video',f"{filename_ext}".mp4)))


@app.route('/background_process_start')
def background_process_start():
    """Task
    """
    global filename, filename_ext, color1, color2, color3
    filename_ext= filename.rsplit(".",1)[0]
    makedirs(join('static','images',filename_ext), exist_ok = True)
    video_path,chunks = video_to_frame_multiprocessing(filename)
    print(chunks)
    extract_images(video_path, filename_ext, chunks[0][0], chunks[0][1],
            color1, color2, color3)
    return redirect(url_for('finalvideo'))

@app.route('/background_process_stop')
def background_process_stop():
    """Exits the program
    TODO: exit the program properly
    """
    exit()

if __name__ == "__main__":
    app.run(debug=True)
