
# Creating the CasualConv1D from the Conv1D of tensorflow
# CasualConv1D is a updation of Conv1d layer of tensorflow by adding a dilation factor in Conv1d
# This will be the basic building layer for our convolutional neural net

class CausalConv1D(tf.layers.Conv1D):
    def __init__(self, filters,
               kernel_size,
               strides=1,
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )
       
    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        return super(CausalConv1D, self).call(inputs)


class TemporalBlock(tf.layers.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.2, 
                 trainable=True, name=None, dtype=None, 
                 activity_regularizer=None, **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )        
        self.dropout = dropout
        self.n_outputs = n_outputs
        self.conv1 = CausalConv1D(
            n_outputs, kernel_size, strides=strides, 
            dilation_rate=dilation_rate, activation=tf.nn.relu, 
            name="conv1")
        self.conv2 = CausalConv1D(
            n_outputs, kernel_size, strides=strides, 
            dilation_rate=dilation_rate, activation=tf.nn.relu, 
            name="conv2")
        self.down_sample = None

    
    def build(self, input_shape):
        channel_dim = 2
        self.dropout1 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        if input_shape[channel_dim] != self.n_outputs:
            # self.down_sample = tf.layers.Conv1D(
            #     self.n_outputs, kernel_size=1, 
            #     activation=None, data_format="channels_last", padding="valid")
            self.down_sample = tf.layers.Dense(self.n_outputs, activation=None)
        self.built = True
    
    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        return tf.nn.relu(x + inputs)


# Creating a temporal convolutional neural network
# This will create required layers of temporal blocks

class TemporalConvNet(tf.layers.Layer):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2,
                 trainable=True, name=None, dtype=None, 
                 activity_regularizer=None, **kwargs):
        super(TemporalConvNet, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            self.layers.append(
                TemporalBlock(out_channels, kernel_size, strides=1, dilation_rate=dilation_size,
                              dropout=dropout, name="tblock_{}".format(i))
            )
    
    def call(self, inputs, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)
        return outputs



# assumption
# 6 physiomarkers
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
num_input = 6 # 6 physiomarkers input 
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
            # gvs = optimizer.compute_gradients(loss_op)
            # for grad, var in gvs:
            #     if grad is None:
            #         print(var)
            # capped_gvs = [(tf.clip_by_value(grad, -.5, .5), var) for grad, var in gvs]
            # train_op = optimizer.apply_gradients(capped_gvs)    
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
