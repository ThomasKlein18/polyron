import numpy as np 
import tensorflow as tf 
import data_provider as d_prov

flags = tf.flags # cmd line FLAG manager for tensorflow
logging = tf.logging # logging manager for tensorflow

flags.DEFINE_string("save_path", None,
    "The path where we'll store checkpoints and summaries.")
flags.DEFINE_string("data_path", "/Users/thomasklein/Projects/Polyron/polyron/mnist/",
    "The path from where to grab training data.")
flags.DEFINE_integer("epochs", 100,
    "The number of epochs for which to train the model.")
flags.DEFINE_integer("batchsize", 128,
    "The batchsize, what did you think this would be?")
    
FLAGS = flags.FLAGS


def model_fn(features, labels, mode, params):

    print("features:", features) # should be shape batchsize x 28 x 28
    features = tf.reshape(features, [-1, 28*28]) # should create shape batchsize x 784
    print("features: ", features)

    layer = features

    for layer_dim in params['layers']:
        layer = tf.layers.dense(layer, layer_dim, activation=tf.nn.tanh)

    logits = tf.layers.dense(layer, params['classes'], activation=None)
    print("logits:", logits) # should have shape batchsize x 10

    # Compute predictions
    predicted_classes = tf.argmax(logits, 1)
    print("predicted_classes:", predicted_classes) # should have shape batchsize x 1

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
    summary_op = tf.summary.merge_all() 

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
            'layers': [64, 32],
            'classes': 10
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

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)