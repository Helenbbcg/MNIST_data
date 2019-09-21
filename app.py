from flask import Flask,request
import cv2
import time
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import record

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b 
sess = tf.Session()
saver=tf.train.Saver() 
saver.restore(sess,'saver/model.ckpt')
    
app = Flask(__name__)
@app.route('/upload',methods=['POST'])
def predict():
    file = request.files['file']
    filename=secure_filename(file.filename)
    im = cv2.imread(filename,cv2.IMREAD_COLOR).astype(np.float32)
    im = cv2.resize(im,(28,28),interpolation=cv2.INTER_CUBIC)
    img_gray = (im - (255 / 2.0)) / 255
    x_img = np.reshape(img_gray , [-1 , 784])
    output=sess.run(y ,feed_dict={x:x_img})
    result = str(np.argmax(output))
    t=int(round(time.time()*1000))
    date=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(t/1000))
    record.recordData(date,filename,result)  
    return "The prediction of your picture is:"+result      
  
if __name__ == '__main__':
    record.createKeySpace()
    app.run()