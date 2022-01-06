import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import requests
import os
import cv2
from colorama import Fore, Style, init
import time

init(convert= True)

model = tf.keras.models.load_model('keras_model.h5')

def identify(url):
  np.set_printoptions(suppress=True)
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  image = Image.open(requests.get(url, stream=True).raw)
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.ANTIALIAS)
  image_array = np.asarray(image)
  normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
  data[0] = normalized_image_array
  prediction = model.predict(data)
  class_list = ["Mask", "No Mask"]

  if prediction[0][0] > 0.7:
    return class_list[0]

  if prediction[0][1] > 0.7:
    return class_list[1]

def identify2(image):
  np.set_printoptions(suppress=True)
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.ANTIALIAS)
  image_array = np.asarray(image)
  normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
  data[0] = normalized_image_array
  prediction = model.predict(data)
  class_list = ["Mask", "No Mask"]

  if prediction[0][0] > 0.7:
    return class_list[0]

  if prediction[0][1] > 0.7:
    return class_list[1]

def camera():
  cam = cv2.VideoCapture(0)
  image = cam.read()[1]
  cv2.imshow("image", image)
  cv2.imwrite("image.png", image)
  cv2.destroyAllWindows()

menu =f'''{Fore.LIGHTBLUE_EX} 
                                         ╔══════════════════════════════╗
                                                  [1] Start
                                                  [2] Exit 
                                         ╚══════════════════════════════╝    {Style.RESET_ALL}
        '''

menu2 =f'''{Fore.LIGHTBLUE_EX} 
                                         ╔══════════════════════════════╗
                                                 [1] URL
                                                 [2] Camera
                                                 [3] Exit
                                         ╚══════════════════════════════╝    {Style.RESET_ALL}
        '''

os.system("cls")
print(menu)
choice = input(f"{Fore.LIGHTBLUE_EX}                                         [!]{Style.RESET_ALL} [1/2] > ")

if choice == "1":

  while True:
    os.system("cls")
    print(menu2)
    choice = input(f"{Fore.LIGHTBLUE_EX}                                         [!]{Style.RESET_ALL} [1/2/3] > ")

    if choice == "1":
      os.system("cls")
      url = input("URL: ")
      print(menu2)
      print(identify(url))
      time.sleep(3)
      os.system("cls")

    if choice == "2":
      os.system("cls")
      camera()
      image = Image.open("image.png")
      print(menu2)
      print(identify2(image))
      time.sleep(3)
      os.system("cls")

    if choice == "3":
      exit()
      
else:
  exit()
