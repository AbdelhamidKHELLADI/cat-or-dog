import os
import glob
import shutil

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img,img_to_array
import numpy as np


def slpit_data(images_path,path_to_save_train,path_to_save_val,path_to_save_test):
    """ This function is used to split the data into train, validation and test sets.
    Args:
        images_path: Path to the images folder.
        path_to_save_train: Path to save the train images.
        path_to_save_val: Path to save the validation images.
        path_to_save_test: Path to save the test images.
    Returns:
        None
    """

    folders = os.listdir(images_path)
    for folder in folders:
        full_path = os.path.join(images_path,folder)
        img=glob.glob(full_path+'/*.jpg')
        train,test = train_test_split(img,test_size=0.2)
        train,val = train_test_split(train,test_size=0.1)
        for i in train:
            path=os.path.join(path_to_save_train,folder)
            #if image is in path_to_save_train folder then skip it
            if not os.path.exists(path):
                os.makedirs(path)
            if i in path:
                continue
            shutil.copy(i,path)
        for i in test:
            path=os.path.join(path_to_save_test,folder)
            if not os.path.exists(path):
                os.makedirs(path)
            if i in path:
                continue
            shutil.copy(i,path)
        for i in val:
            path=os.path.join(path_to_save_val,folder)
            if not os.path.exists(path):
                os.makedirs(path)
            if i in path:
                continue
            shutil.copy(i,path)




def generators(train_path,val_path,test_path):
    """
    This function is used to create the generators for the train, validation and test sets.
    Args:
        train_path: Path to the train images.
        val_path: Path to the validation images.
        test_path: Path to the test images.
    Returns:
        train_gen: Train generator.
        val_gen: Validation generator.
        test_gen: Test generator.
    """
    gen=ImageDataGenerator(rescale=1./255)
    train_gen=gen.flow_from_directory(train_path,target_size=(256,256),batch_size=64,class_mode='categorical',shuffle=True,color_mode='rgb')
    val_gen=gen.flow_from_directory(val_path,target_size=(256,256),batch_size=64,class_mode='categorical',shuffle=True,color_mode='rgb')
    test_gen=gen.flow_from_directory(test_path,target_size=(256,256),batch_size=64,class_mode='categorical',shuffle=True,color_mode='rgb')
    return train_gen,val_gen,test_gen


def predict_with_model(model,image_path):
    img=load_img(image_path,target_size=(256,256))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,axis=0)
    pred=model.predict(img)
    pred=np.argmax(pred)
    if pred==0:
        pred='cat'
    else:
        pred='dog'
    return pred