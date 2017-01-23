"""Making an mnist dataset classifier using tensorflow with the help of sentdex tutorials
Epoch 0 completed out of  10 loss :  1616054.85604
Epoch 1 completed out of  10 loss :  393365.490524
Epoch 2 completed out of  10 loss :  209800.33265
Epoch 3 completed out of  10 loss :  124373.578312
Epoch 4 completed out of  10 loss :  75902.4687478
Epoch 5 completed out of  10 loss :  47859.0475696
Epoch 6 completed out of  10 loss :  33407.6449745
Epoch 7 completed out of  10 loss :  24222.0732288
Epoch 8 completed out of  10 loss :  20512.9660247
Epoch 9 completed out of  10 loss :  19604.4842276
Accuracy: 0.953
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# number of nodes in hidden layer 1
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
batch_size = 100

x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = { 'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = { 'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = { 'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = { 'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
    'biases':tf.Variable(tf.random_normal([n_classes]))}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # feedforward + back props
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x , epoch_y = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost],feed_dict = {x:epoch_x,y:epoch_y})
                epoch_loss += c

            print  'Epoch',epoch,'completed out of ',hm_epochs,'loss : ',epoch_loss
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print 'Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels})

train_neural_network(x)
