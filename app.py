from flask import Flask
import cv2
import numpy as np
import tensorflow as tf
app = Flask(__name__)

@app.route('/')
def predict():
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run() 
    saver=tf.train.Saver()  
    im = cv2.imread('MNIST_data/images/2.png',cv2.IMREAD_COLOR).astype(np.float32)
    im = cv2.resize(im,(28,28),interpolation=cv2.INTER_CUBIC)
    img_gray = (im - (255 / 2.0)) / 255
    x_img = np.reshape(img_gray , [-1 , 784])
    output=sess.run(y ,feed_dict={x:x_img})
    result = np.argmax(output)
    return "The prediction of your picture is: "+ str(result)
    with tf.Session().as_default()as sess:
        saver.restore(sess,'saver/model.ckpt')
  
if __name__ == '__main__':
    app.run()