import input_data
import tensorflow as tf

tf.app.flags.DEFINE_string("log_dir","softmax_logs","Directory for saving the summaries")
tf.app.flags.DEFINE_integer("max_it",1000,"Number of iterations to run")
tf.app.flags.DEFINE_integer("batch",100,"Batch size")
FLAGS = tf.app.flags.FLAGS

print "Logging summaries to:",FLAGS.log_dir

mnist = input_data.read_data_sets("data/", one_hot=True)

print "Training images size :",mnist.train.images.shape
print "Training labels size :",mnist.train.labels.shape
print "Testing images size :",mnist.test.images.shape
print "Testing labels size :",mnist.test.labels.shape

# Creating name scope in the graph for the input placeholders
with tf.name_scope("Input_Data") as scope:
  x = tf.placeholder(tf.float32, [None,784], name="x")
  y_ = tf.placeholder(tf.float32, [None,10], name="y")


# Creating variables for storing weights and biases
W = tf.Variable(tf.zeros([784,10]), name="weights")
b = tf.Variable(tf.zeros([10]), name="bias")

# Creating name scope in the graph for the applying softmax regression
with tf.name_scope("Wx_b") as scope:
  y = tf.nn.softmax(tf.matmul(x,W)+b)


# Creating histogram summaries for weights, biases and output labels
w_hist = tf.histogram_summary("weights", W)
b_hist = tf.histogram_summary("biases", b)
y_hist = tf.histogram_summary("y", y)

# Creating name scope in the graph for cross entropy ,i.e, performance measure
with tf.name_scope("Xentropy") as scope:
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  ce_summ = tf.scalar_summary("cross entropy", cross_entropy)


# Creating name scope in the graph for training step using standard gradient descent optimizer minimizing the cross entropy defined above
with tf.name_scope("Train") as scope:
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# Creating name scope in the graph for testing step
with tf.name_scope("Test") as scope:
  correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  accuracy_summary = tf.scalar_summary("accuracy", accuracy)


# Merging all the summaries created above
merged = tf.merge_all_summaries()

# Initializing the variables
init = tf.initialize_all_variables()

# Creating the session for the graph
sess = tf.Session()

# Initializing the summary writer for the graph to the specified directory
writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph_def)

# Running the session of the graph for the variable initialization defined above
sess.run(init)


for i in range(FLAGS.max_it):
  if i % 10 == 0:  # Record summary data, and the accuracy

    # Getting values for the placeholders
    feed = {x: mnist.test.images, y_: mnist.test.labels}

    # Running the session of the graph for calculating the Accuracy and summaries by the feeding the data fetched above
    result = sess.run([merged, accuracy], feed_dict=feed)

    summary_str = result[0]
    acc = result[1]

    # Add summary to the summary writer after every 10 steps
    writer.add_summary(summary_str, i)

    print("Accuracy at step %s: %s" % (i, acc))
  else:
    # Getting next 10 data points for training
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch)
      feed = {x: batch_xs, y_: batch_ys}

    # Running the session of the graph for Gradient Descent Optimization with data points fetched above
      sess.run(train_step, feed_dict=feed)

print "=========================================================="
print "Final Evaluation"

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
