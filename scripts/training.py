import tensorflow as tf
from model import TemporalConvNet, TemporalBlock, CausalConv1D


# assumption
# 9 physiomarkers
# 6 hour of high frequency data in seconds unit

# Either patient is suffering or not suffering from the sepsis disease
# kernel size and number of hidden layers will be changed accordingly after calculating the metrics on different architecture
# Best training parameters will be based on the model architecture which is performing best on different metrics

# Training Parameters
learning_rate = 0.001
batch_size = 80
total_batch = # depending on the dataset from part1
print("Number of batches per epoch:", total_batch)
training_steps = 3000

# Network Parameters
num_input = 9 # 9 physiomarkers input 
timesteps = 6 * 60 * 60 # timesteps
num_classes = 2 # 1 or 0
dropout = 0.1
kernel_size = 8
levels = 6
nhid = 20 # hidden layer num of features


def graph():
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(10)
        # tf Graph input
        X = tf.placeholder("float", [None, timesteps, num_input])
        Y = tf.placeholder("float", [None, num_classes])
        is_training = tf.placeholder("bool")
        
        # Define weights
        logits = tf.layers.dense(
            TemporalConvNet([nhid] * levels, kernel_size, dropout)(
                X, training=is_training)[:, -1, :],
            num_classes, activation=None, 
            kernel_initializer=tf.orthogonal_initializer()
        )
        prediction = tf.nn.softmax(logits)
    
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=Y))
        
        with tf.name_scope("optimizer"):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)   
            train_op = optimizer.minimize(loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        print("All parameters:", np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.global_variables()]))
        print("Trainable parameters:", np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))

def training():
    graph()
    with tf.Session(graph=graph, config=config) as sess:
    # Run the initializer
        sess.run(init)
        for step in range(1, training_steps+1):
            batch_x, batch_y = # data obtained from the part 1
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_training: True})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={
                    X: batch_x, Y: batch_y, is_training: False})
                # Calculate accuracy for the test data
                test_len = # number of test examples
                test_data = # physiomarkers of test examples
                test_label = # output whether patient is suffering from sepsis or not
                val_acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label, is_training: False})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                    "{:.4f}".format(loss) + ", Training Accuracy= " + \
                    "{:.3f}".format(acc) + ", Test Accuracy= " + \
                    "{:.3f}".format(val_acc))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_path = saver.save(sess, "/tmp/model.ckpt")
                    print("Model saved in path: %s" % save_path)
        print("Optimization Finished!")