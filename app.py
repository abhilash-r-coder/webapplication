from flask import Flask, render_template, request, redirect, url_for, session
import re
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
from flask_material import Material
import cv2
import numpy as np
from skimage.feature import hog
from keras.models import Sequential
from tensorflow.keras.models import Sequential
from keras.layers import Activation, Dense
import pickle
from tensorflow.keras.models import load_model
UPLOAD_FOLDER = 'static/uploads/'

# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg


app = Flask(__name__)
Material(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
img_height = 300
img_width = 300
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Enter your database connection details below

@app.route('/')
def index():
    return render_template("login.html")

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():

    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/',methods=['GET', 'POST'])
def login():
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        
                # If account exists in accounts table in out database
        if username=="abhi" and password=="abhilash":
            # Create session data, we can access this data in other routes
            # Redirect to home page
            return render_template('index.html')
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)



def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    hog_features, _ = hog(gray_image, orientations=8, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True)
    return hog_features.reshape(1, -1)

@app.route('/upload_image',methods=["POST"])
def upload_image():
	
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    print(file)
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        input_image_path =path  # Replace with the actual image path
        input_image = cv2.imread(input_image_path)
        input_hog_features = preprocess_image(input_image)

        # Create the CNN model
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(23328,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model = load_model("SignatureForgery.h5")


        prediction = model.predict(input_hog_features)
        prediction_class = "Forged" if prediction[0][0] > 0.5 else "Genuine"
        print(f"Prediction: {prediction_class}")



        return render_template('contact.html',aclass=prediction_class,res=1)



if __name__ == '__main__':
	app.run(debug=True)
