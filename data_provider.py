"""
This file contains the input functions used by the estimators to feed MNIST data.
"""
import tensorflow as tf
import os



def read_dataset(path, mode, batch_size, repeat, seq_length, seq_width, out_dtype, tfrecord_dtype):
    """
    Reads data from .tfrecords-file, decodes it and returns the dataset as a
    tf.data.TFRecordDataset.
    This probably only works for data that was generated using the export_data-script.
    batch_size      int, the batch_size used during training
    num_epochs      int, the number of training epochs
    seq_length      int, the length of the sequences (eg an mnist-image is 784 long)
    seq_width       int, the width of the sequences (dimensionality of the data)
    """

    def _parse_function(example_proto):
        """
        The function that is used to parse the data in the tfrecords-file.
        Both the MNIST-data as well as the AR-data have the column names
        "raw_label" and "raw_sequence", so there is no need for a second parse
        function here.
        If you want to extend this experiment to other datasets, you might need
        to write a different parse function and make read_dataset accept this as
        an argument.
        """
        features = {"raw_sequence": tf.FixedLenFeature([], tf.string, default_value=""),
                    "raw_label": tf.FixedLenFeature([], tf.string, default_value="")}
        parsed_features = tf.parse_single_example(example_proto, features)

        # NOTE: the reason that this will be an int32 is weird and hidden;
        # retrieve sequence
        seq = tf.decode_raw(parsed_features["raw_sequence"], tfrecord_dtype)
        seq.set_shape(seq_length*seq_width)
        seq = tf.reshape(seq, [seq_length, seq_width])
        seq = tf.cast(seq, out_dtype)

        # retrieve labels [i.e. last and to be predicted element of sequence]
        label = tf.decode_raw(parsed_features["raw_label"], tfrecord_dtype)
        label.set_shape(1)
        label = tf.cast(label, tf.int32)

        return seq, label

    training_path = os.path.join(path, mode+'.tfrecords')
    training_dataset = tf.data.TFRecordDataset(training_path)
    training_dataset = training_dataset.map(_parse_function)
    training_dataset = training_dataset.shuffle(64000)
    training_dataset = training_dataset.batch(batch_size, drop_remainder=True)

    # Pro Tip: Don't use repeat. I learned that the hard way:
    # If you start the training anew for each epoch, by calling train with the input function, 
    # the data is reshuffled. If you use repeat, that does not happen, the data is just repeated a bunch of times in that order.
    # This is usually not the behaviour you want, and it can have suprisingly strong negative effects on training.
    #if(repeat):
    #    training_dataset = training_dataset.repeat()

    training_dataset = training_dataset.prefetch(1)
    return training_dataset 


def input_fn(path, config, mode, repeat):
    """
    Core input function. Calls the read_dataset function with the appropriate parameters.
    path   =  the path to the tfrecords-files
    mode   =  'train', 'test' or 'validation'
    config =  a configuration object
    repeat =  whether to repeat the dataset or not
    """
    return read_dataset(path, mode, config.batchsize, repeat, 
        seq_length=config.seq_length, seq_width=config.seq_width, 
        out_dtype=config.dtype, tfrecord_dtype=config.tfrecord_dtype)
    

def train_input_fn(path, config):
    """
    input function used for training data
    These functions are only convenience wrappers for the input_fn above.
    """
    return input_fn(path, config, 'train', True)


def validation_input_fn(path, config):
    """
    input function used for validation data (used to check for overfitting)
    These functions are only convenience wrappers for the input_fn above.
    """
    return input_fn(path, config, 'validation', False)


def test_input_fn(path, config):
    """
    input function used for test data (to evaluate final performance)
    These functions are only convenience wrappers for the input_fn above.
    """
    return input_fn(path, config, 'test', False)