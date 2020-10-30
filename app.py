
import os
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from flask import Flask, request, render_template,send_from_directory,send_file
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 
import cv2                
import matplotlib.pyplot as plt  
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
import logging



logger = logging.getLogger(__name__)
f_handler = logging.FileHandler('file.log')
f_format = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)   
# define function to load train, test, and validation datasets
def load_dataset(path):
    try:

        data = load_files(path)
        dog_files = np.array(data['filenames'])
        dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
        return dog_files, dog_targets
    
    except Exception as e:
        
        logger.error("Error while loading dataset."+str(e))
        return str(e)
    

APP_ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))

saved_models_path=os.path.join(APP_ROOT_FOLDER, '{}'.format("saved_models/weights.best.Resnet50.hdf5"))
 
train_files, train_targets = load_dataset(os.path.join(APP_ROOT_FOLDER, '{}'.format("dogImages/train")))
valid_files, valid_targets = load_dataset(os.path.join(APP_ROOT_FOLDER, '{}'.format("dogImages/valid")))
test_files, test_targets = load_dataset(os.path.join(APP_ROOT_FOLDER, '{}'.format("dogImages/test")))


# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

#Train the model.import random
import random
random.seed(8675309)

# load filenames in shuffled human dataset

# extract pre-trained face detector
#face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    try:

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0
    except Exception as e:
        logger.error("Error while detecting face."+str(e))
        #Error when detecting face
        return False

# define ResNet50 model
ResNet50_DogDetector_model = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

   



def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_DogDetector_model.predict(img))

    

    ### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    try:

        prediction = ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151))
    except Exception as e:
        logger.error("Error while detecting dog image."+str(e))
        #Error while detecting dog Image
        return False

    ## Obtain bottleneck features from another pre-trained CNN and train the model
def trainmodel():
    bottleneck_features_Resnet = np.load('bottleneck_features/DogResnet50Data.npz')
    train_Resnet50 = bottleneck_features_Resnet['train']
    valid_Resnet50 = bottleneck_features_Resnet['valid']
    test_Resnet50 = bottleneck_features_Resnet['test']

    ### TODO: Define your architecture.
    Resnet50_model = Sequential()

    Resnet50_model.add(Flatten(input_shape=train_Resnet50.shape[1:]))
    Resnet50_model.add(Dense(500, activation='relu'))
    Resnet50_model.add(Dropout(0.5))
    Resnet50_model.add(Dense(133, activation='softmax'))

    Resnet50_model.summary()

    """### (IMPLEMENTATION) Compile the Model"""

    ### TODO: Compile the model.
    Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    """### (IMPLEMENTATION) Train the ModelTrain your model in the co"""
    
    #saved_models_path=os.path.join(APP_ROOT_FOLDER, '{}'.format("saved_models/weights.best.Resnet50.hdf5"))
    checkpointer = ModelCheckpoint(filepath=saved_models_path ,verbose=1, save_best_only=True)

    Resnet50_model.fit(train_Resnet50, train_targets, 
            validation_data=(valid_Resnet50, valid_targets),
            epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
    ### TODO: Calculate classification accuracy on the test dataset.
   
    
    Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
    return test_accuracy

## Load the model weights with the best validation loss.
def load_ResNet50Model():
    Resnet50_model = Sequential()

    Resnet50_model.add(Flatten(input_shape=(1,1,1,2048)))
    Resnet50_model.add(Dense(500, activation='relu'))
    Resnet50_model.add(Dropout(0.5))
    Resnet50_model.add(Dense(133, activation='softmax'))

    Resnet50_model.summary()

    """### (IMPLEMENTATION) Compile the Model"""

    ## Compile the model.
    Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    Resnet50_model.load_weights(saved_models_path)
    return Resnet50_model
    ### function that takes a path to an image as input
    ### and returns the dog breed that is predicted by the model.
 
def Resnet_predict_breed(img_path,Resnet50_model):
    
    
    # extract bottleneck features
    bottleneck_feature = ResNet50(weights='imagenet', include_top=False,pooling="avg").predict(preprocess_input(path_to_tensor(img_path)))
    bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
    bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
    # obtain predicted vector
    
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    #Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]
        
    # return dog breed that is predicted by the model
        
    return dog_names[np.argmax(predicted_vector)]

   
#Final algorithm for predicton

def dogOrHuman(img_path,Resnet50_model):
    

    dogorhuman=""
    returnstr=""
    if dog_detector(img_path):
        prediction=Resnet_predict_breed(img_path,Resnet50_model)
        #returnstr="Hello dog! You look like " +(Resnet_predict_breed(img_path,Resnet50_model)+"\n\n\n")
        dogorhuman,returnstr="Hello dog! You look like "+str(prediction).replace("_"," ") ,prediction
      
        
    elif face_detector(img_path):
        prediction=Resnet_predict_breed(img_path,Resnet50_model)
        #returnstr="Hello human! You look like " +(Resnet_predict_breed(img_path,Resnet50_model)+"\n\n\n")
        dogorhuman,returnstr=("Hello human! You look like "+str(prediction).replace("_"," ") ,prediction)
      
    else:
        dogorhuman="Oops!This image looks like alien to me"
        
    return dogorhuman,returnstr




#Predict input image
def predictImage(input_image,Resnet50_model):
    try:

        dogorhuman,result = dogOrHuman(input_image,Resnet50_model)

        return dogorhuman,result
    except Exception as e:
        raise Exception(e)

app = Flask(__name__)
model = None
rootpath=os.path.join(APP_ROOT_FOLDER)
target = os.path.join(APP_ROOT_FOLDER, '{}'.format("sample_images"))
inputImage=""
Resnet50_model=load_ResNet50Model()

#Flask API GET function on initial application load 

@app.route("/")
def index():
    print("entered index")
    
    try:

        #uploadhtmlpath = os.path.join(APP_ROOT_FOLDER, '{}'.format("templates/upload.html"))
        return render_template("upload.html")
    except Exception as e:
        print("exception",e)
        logger.error("Error when calling GET flask API for initial load."+str(e))
        return str(e)
   

#Flask API POST request for predicring on input image
@app.route('/predict', methods=['POST'])
def predict():
    try:
        imagename=""
        if request.method == "POST":
            if request.files:
                image=request.files["image"]
                #Save input image to local folder
                destination="/".join([target, image.filename])
                image.save(destination)
            
        sample_images = glob("./sample_images/*")
        
        for img in sample_images:
            #Get input image from local folder
            input_image=img

         #Final result   
        dogorhuman,result= predictImage(input_image,Resnet50_model)


        #Path for dog category names
        dogcategories = os.path.join(APP_ROOT_FOLDER, '{}'.format("dogImages/train"))
        
        #Loop over all dog categories
        for dogname in os.listdir(dogcategories):
            lastindexofdot=dogname.rfind('.')
            substringtosearch=dogname[lastindexofdot+1:]
            
            isdogname_match=substringtosearch.strip()==str(result).strip()

            # When we find predicted result dog name in our train folder
            #Go into that folder and pick one image path
            if isdogname_match:
                dogcategory_images = os.path.join(APP_ROOT_FOLDER, '{}'.format("dogImages/train/"+dogname))
                for dog_image in os.listdir(dogcategory_images):
                    imagenamepath=os.path.join(APP_ROOT_FOLDER, '{}'.format("dogImages/train/"+dogname+"/"+dog_image))
                    indextosplit=str(imagenamepath).rfind('dogImages')
                    imagename=str(imagenamepath)[indextosplit:]
                    break
        if not imagename:
           
            # imagename="alien_image/alien_image.jpg"
            imagenamepath=os.path.join(APP_ROOT_FOLDER, '{}'.format("alienImage/alien_image.jpg/"))
            indextosplit=str(imagenamepath).find('alienImage')
            imagename=str(imagenamepath)[indextosplit:]
            
                
        #remove the input images from the local folder 

        for filename in os.listdir(target):
            try:
                file_path = os.path.join(target, filename)
                
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error('Failed to delete.'+str(e))

                
       
       
    
    except Exception as e:
        logger.error("Error while making prediction."+str(e))
        return str(e)
          
    return render_template("complete.html", image_name=imagename,value=dogorhuman)
    

#Return compalte html page with predictions
@app.route('/<path:imagename>')  
def send_image(imagename):  
    try:
        
        return send_from_directory(rootpath, imagename)
    except Exception as e:
        logger.error('Error when calling send image flask API.'+str(e))
        return str(e)



if __name__ == '__main__':
    # Inital point of application
    
    
    app.run(host='0.0.0.0', port=80)
    

    
    
    


