#ensures that tensorflow Lite is used as the backend
from tflite_runtime.interpreter import Interpreter 
from os import listdir
from PIL import Image
import numpy as np
import time

def standardize_me1(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

# make it between 0 and 1    
def rescale_me1(image):
    image = np.array(image / 255., dtype=np.float32)
    return image

# normalize between min and max    
def normalize_me1(image):
    #image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
    image *= 1/image.max()
    return image


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  # fit the model input dimensions
  image = np.expand_dims(image, axis=2)
  input_tensor[:, :] = image    

    
def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))
  scale, zero_point = output_details['quantization']
  #output = scale * (output.astype(np.float32) - zero_point)
  # no quantization therefore scale is not used
  output = output.astype(np.float32) - zero_point
  return output    


def cust_predict(file_dir, model_path):

    interpreter = Interpreter(model_path)
    #input_details = interpreter.get_input_details()
    #output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    print("Input Shape (", width, ",", height, ")")
    # PNG images are in the '1' subfolder
    file_dir = file_dir + '/1/'
    prediction_list = []
    
    for f in sorted(listdir(file_dir)):
      #print(f)
      image = Image.open(file_dir + f)
      image = image.resize((width, height))
      # convert image to numpy array
      image = np.asarray(image, dtype=np.float32)

      # rescale values to (0;1)
      image = rescale_me1(image)
      # normalize min and max values
      image = normalize_me1(image)
      #image = standardize_me1(image)
    	
      # Classify the image.
      time1 = time.time()
      prob = classify_image(interpreter, image)
      time2 = time.time()
      classification_time = np.round(time2-time1, 3)
      print(f'{prob:.3f}' + "\t" + f + "\t" + str(classification_time), " seconds.")
      prediction_list.append((f'{prob:.3f}', f))

    return prediction_list

if __name__ == "__main__":
    if len(sys.argv) != 1:
        png_dir_1 = sys.argv[1]
        model_path_1 = sys.argv[2]
    else:
        png_dir_1 = input("enter path of png file directory to be used: ")
        model_path_1 = input("enter path of .h5 model to be used: ")
    print(cust_predict(png_dir_1, model_path_1))
