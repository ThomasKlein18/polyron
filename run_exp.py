import numpy as np 
import tensorflow as tf 
import data_provider as d_prov

flags = tf.flags # cmd line FLAG manager for tensorflow
logging = tf.logging # logging manager for tensorflow

flags.DEFINE_string("save_path", None,
    "The path where we'll store checkpoints and summaries.")
flags.DEFINE_string("data_path", "/Users/thomasklein/Projects/Polyron/polyron/mnist/",
    "The path from where to grab training data.")
flags.DEFINE_integer("epochs", 20,
    "The number of epochs for which to train the model.")
flags.DEFINE_integer("batchsize", 32,
    "The batchsize, what did you think this would be?")
flags.DEFINE_integer("degree", 5,
    "The degree of the polynomials.")
flags.DEFINE_string("mode", "poly",
    "The activation function  mode. Choose from: relu, tanh, poly.")
    
FLAGS = flags.FLAGS


def mlp_layer(x, neurons_out, degree, varscope, mode):
    """
    Creates a vanilla feed-forward layer that takes intake as an input and has neurons_out output neurons.
    
    intake      = the input tensor, e.g. [batchsize, 512]
    neurons_out = the number of output neurons. If this is the last layer, the number of classes.
    degree      = the degree of the polynomial to be used as activation function
    varscope    = the variable scope to be used (just some unique string)
    mode        = which activation mode to use. relu and tanh are simply applied, poly requires extra variables
    """
    shapes = x.get_shape().as_list()
    batchsize = shapes[0]
    neurons_in = shapes[1]

    with tf.variable_scope(varscope):
    
        initializer = tf.random_normal_initializer(stddev = neurons_in**(-1/2))
        weights = tf.get_variable(name="weights", initializer=initializer([neurons_in, neurons_out]), dtype=tf.float32)
        # bias might not be necessary since the polynomial has a y-axis-intercept
        bias = tf.get_variable("bias", initializer=tf.constant(0.0, shape = [neurons_out]))
        
        logits = tf.matmul(x, weights)+bias

        if mode == 'poly':

            coefficients = tf.get_variable("coefficients",initializer=initializer([neurons_out,degree]), dtype=tf.float32)

            sum_list = []
            for b in range(batchsize):
                inner_sum = tf.constant(np.zeros((neurons_out)), dtype=tf.float32)
                for i in range(degree):
                    inner_sum += tf.multiply(tf.reshape(tf.math.pow(logits[b,:],tf.constant(i, dtype=tf.float32)), [neurons_out]), tf.reshape(coefficients[:,i], [neurons_out]))
                sum_list.append(inner_sum)
            res = tf.stack(sum_list)
            print("resulting shape: ",res)
            return res # summe
        
        elif mode == 'relu':
            return tf.nn.relu(logits)
        elif mode == 'tanh':
            return tf.nn.tanh(logits)
        else:
            raise NotImplementedError("Mode not recognized, use 'poly', 'relu' or 'tanh'.")


def model_fn(features, labels, mode, params):

    features = tf.reshape(features, [-1, 28*28]) # should create shape batchsize x 784, in case images were supplied differently
    features = features / 255 - 0.5

    layer = features
    #import sys
    #layer = tf.Print(layer, [layer], summarize=-1)

    for idx, layer_dim in enumerate(params['layers']):
        #layer = tf.layers.dense(layer, layer_dim, activation=get_activation_function(FLAGS.mode))
        layer = mlp_layer(layer, layer_dim, params['degree'], "layer"+str(idx), params['mode'])

    logits = tf.layers.dense(layer, params['classes'], activation=None)

    # Compute predictions
    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    
def main(_):

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.save_path,
        params={
            'layers': [32, 16],
            'classes': 10,
            'degree': FLAGS.degree,
            'mode': FLAGS.mode
        })

    class Config:
        batchsize = FLAGS.batchsize
        tfrecord_dtype = tf.uint8
        dtype = tf.float32
        seq_length = 784
        seq_width = 1

    config = Config()

    
    for epoch in range(FLAGS.epochs):

        # Train the Model.
        classifier.train(
            input_fn=lambda:d_prov.train_input_fn(FLAGS.data_path, config),
            steps=64000/FLAGS.batchsize) # assuming MNIST/fashion MNIST for now

        #Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=lambda:d_prov.validation_input_fn(FLAGS.data_path, config),
            steps=100,
            name="validation")

        print('\nValidation set accuracy after epoch {}: {accuracy:0.3f}\n'.format(epoch+1,**eval_result))

    eval_result = classifier.evaluate(
        input_fn=lambda:d_prov.test_input_fn(FLAGS.data_path, config),
        name="test"
    )
    
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # ---------- printing variables ------------ #

    #vars = tf.trainable_variables()
    #print(vars)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)