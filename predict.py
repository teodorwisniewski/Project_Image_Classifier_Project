# imports
from argparse import ArgumentParser
import csv
import json
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


# user defined functions
def process_image(image):
    image_size = 224
    # from integers to floats
    image = tf.cast(image, tf.float32) 
    image = tf.image.resize(image, (image_size, image_size))
    # normalizing values
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k: int):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    inputs = np.expand_dims(processed_test_image, axis=0)
    # getting probability distribution
    y_hat = model.predict(inputs)
    # sorting output probability values
    indexes = np.argsort(y_hat)[0,::-1]
    y_hat = y_hat[0,indexes]
    return y_hat[:top_k], indexes[:top_k].astype(np.str)


if __name__ == "__main__":
    
    
    # Processing inputs
    parser = ArgumentParser(description="My Image Classifier")
    # positional arguments
    parser.add_argument('image_path', help="the path to the image")
    parser.add_argument('model', help="The model saved in the .h5 format")
    # optional arguments
    parser.add_argument('--top_k', default="5", help="Number of classes we want to see", type=int)
    parser.add_argument('--category_names', default=None, help="Json file that allows to map numbers to classes")
    args = parser.parse_args()
    
    
    # Loading data and the model
    if args.category_names is None:
        with open('label_map.json', 'r') as f:
            class_names = json.load(f)
    else:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)    




    saved_keras_model_filepath = args.model
    reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath
                                                      ,custom_objects={'KerasLayer':hub.KerasLayer})
    
# input processing and model predicitons
    image_path = args.image_path
    im = Image.open(image_path)
    test_image = np.asarray(im)
    
    probs, classes = predict(image_path, reloaded_keras_model, args.top_k)
    
    labels = [str(int(x)+1) for x in list(classes)]
    
    
    # printing results on the screen
    print("The predictions in the most likely order: ",[class_names.get(key) for key in labels])
    
    print("Corresponding values Probabilities:", probs)
    


