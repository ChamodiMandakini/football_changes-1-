# TODO: proper configuration rename the file as well to init
from flask import Flask
from os.path import abspath, join
from os import getcwd


# configure the upload folder here
UPLOAD_FOLDER = abspath(join(getcwd(), 'static', 'upload'))


app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# setting up upload
# makesure run python script to create folders and cleanup
