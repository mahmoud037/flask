from flask import Flask,request
import tensorflow as tf
import base64
import numpy as np
from PIL import Image
import OS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
new_model = tf.keras.models.load_model("best_model.h5",compile=False)




app = Flask(__name__)

@app.route('/test')
def hello():
    return ('hello from s flask')

@app.route('/api',methods=['POST'])
def index():
       result=''
       inputchar = request.get_data()
       imgdata = base64.b64decode(inputchar)
       filename = 'somthing.jpg'  
       with open(filename, 'wb') as f:
        f.write(imgdata)
       
       fo=open('labels.txt','r')

       labels=fo.readlines()
       
       test_img = Image.open('somthing.jpg')

       test_img = test_img.resize((256, 256))

       test_img=np.array(test_img)

       test_img=np.expand_dims(test_img,axis=0)

       print(test_img.shape)

       prediction=new_model.predict(test_img,verbose=0)

       result=labels[np.argmax(prediction)]
       
       return (result)
