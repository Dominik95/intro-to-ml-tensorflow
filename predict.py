import sys
import time 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from PIL import Image
import argparse
import json
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

image_size = 224
batch_size = 32

def process_image(input_image):
    image = tf.convert_to_tensor(input_image)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32)
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    image = np.asarray(im)
    processed_image = process_image(image)
    expanded_image = np.expand_dims(processed_image, axis=0)

    ps = model.predict(expanded_image)
    
    top_k_values, top_k_indices = tf.nn.top_k(ps, k=top_k)
    
    top_k_values = top_k_values.numpy()
    top_k_indices = top_k_indices.numpy()
    
    return top_k_values, top_k_indices

def main():
    parser = argparse.ArgumentParser(description='Predict flower images...')
    parser.add_argument('path_to_image', action='store', help='Input the path to the image you want to make a prediction on')
    parser.add_argument('model', action='store', help='Input the model you want to make a prediction with')
    parser.add_argument('--top_k', action='store', default=5, type=int, help='Return the top K most likely classes')
    parser.add_argument('--category_names', action='store', help='Path to a JSON file mapping labels to flower names')
    
    args = parser.parse_args()
    print('path_to_image  = {!r}'.format(args.path_to_image))
    print('model          = {!r}'.format(args.model))
    print('top_k          = {!r}'.format(args.top_k))
    print('category_names = {!r}'.format(args.category_names))
    
    image_path = args.path_to_image
    model = load_model(args.model ,custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
    model.summary()
    top_k = args.top_k
    if (args.category_names != None):
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
    
    probs, classes = predict(image_path, model, top_k)
    
    print(probs)
    fig_classes = []
    if (args.category_names != None):
        for idx in classes[0]:
            fig_classes.append(class_names[str(idx+1)])
        print(fig_classes)    
    else:
        print(classes)
    
    #python predict.py ./test_images/wild_pansy.jpg 1656511663.h5 --category_names label_map.json
    
if __name__ == "__main__":
    main()

