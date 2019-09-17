from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import argparse
#import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import cv2
import numpy as np
#FLAGS = None


# Import data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
    
# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
  
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('The training accuracy is:')
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                  y_: mnist.test.labels}))
saver=tf.train.Saver() 

def prediction(image_name):
    im = cv2.imread(image_name,cv2.IMREAD_COLOR).astype(np.float32)
    im = cv2.resize(im,(28,28),interpolation=cv2.INTER_CUBIC)
    img_gray = (im - (255 / 2.0)) / 255
    x_img = np.reshape(img_gray , [-1 , 784])
    print(x_img)
    global sess
    output=sess.run(y ,feed_dict={x:x_img})
    result = np.argmax(output)
    print('y =: ', '\n',output)
    print('The prediction is:',result)
    return str(result)
    with tf.Session().as_default()as sess:
        saver.restore(sess,'saver/model.ckpt')
    

if __name__ == '__main__':
    prediction('MNIST_data/images/2.png')
    '''parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)'''

  