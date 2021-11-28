# Docs

https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model

They built an exhaustive and useful tutorial -> https://www.tensorflow.org/guide/saved_model

From the docs:

### Save

```py
# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)
```

### Restore

```py
tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
```

# Tensorflow 2

This is still beta so I'd advise against for now. If you still want to go down that road here is the [`tf.saved_model` usage guide](https://www.tensorflow.org/beta/guide/saved_model)

# Tensorflow < 2

## `simple_save`

Many good answer, for completeness I'll add my 2 cents: **[simple_save](https://www.tensorflow.org/programmers_guide/saved_model)**. Also a standalone code example using the `tf.data.Dataset` API.

Python 3 ; Tensorflow **1.14**

```py
import tensorflow as tf
from tensorflow.saved_model import tag_constants

with tf.Graph().as_default():
    with tf.Session() as sess:
        ...

        # Saving
        inputs = {
            "batch_size_placeholder": batch_size_placeholder,
            "features_placeholder": features_placeholder,
            "labels_placeholder": labels_placeholder,
        }
        outputs = {"prediction": model_output}
        tf.saved_model.simple_save(
            sess, 'path/to/your/location/', inputs, outputs
        )
```

Restoring:

```py
graph = tf.Graph()
with restored_graph.as_default():
    with tf.Session() as sess:
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
            'path/to/your/location/',
        )
        batch_size_placeholder = graph.get_tensor_by_name('batch_size_placeholder:0')
        features_placeholder = graph.get_tensor_by_name('features_placeholder:0')
        labels_placeholder = graph.get_tensor_by_name('labels_placeholder:0')
        prediction = restored_graph.get_tensor_by_name('dense/BiasAdd:0')

        sess.run(prediction, feed_dict={
            batch_size_placeholder: some_value,
            features_placeholder: some_other_value,
            labels_placeholder: another_value
        })
```

# Standalone example

**[Original blog post](http://vict0rsch.github.io/2018/05/17/restore-tf-model-dataset/)**

The following code generates random data for the sake of the demonstration.

1. We start by creating the placeholders. They will hold the data at runtime. From them, we create the `Dataset` and then its `Iterator`. We get the iterator's generated tensor, called `input_tensor` which will serve as input to our model.
2. The model itself is built from `input_tensor`: a GRU-based bidirectional RNN followed by a dense classifier. Because why not.
3. The loss is a `softmax_cross_entropy_with_logits`, optimized with `Adam`. After 2 epochs (of 2 batches each), we save the "trained" model with `tf.saved_model.simple_save`. If you run the code as is, then the model will be saved in a folder called `simple/` in your current working directory.
4. In a new graph, we then restore the saved model with `tf.saved_model.loader.load`. We grab the placeholders and logits with `graph.get_tensor_by_name` and the `Iterator` initializing operation with `graph.get_operation_by_name`.
5. Lastly we run an inference for both batches in the dataset, and check that the saved and restored model both yield the same values. They do!

Code:

```py
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


def model(graph, input_tensor):
    """Create the model which consists of
    a bidirectional rnn (GRU(10)) followed by a dense classifier

    Args:
        graph (tf.Graph): Tensors' graph
        input_tensor (tf.Tensor): Tensor fed as input to the model

    Returns:
        tf.Tensor: the model's output layer Tensor
    """
    cell = tf.nn.rnn_cell.GRUCell(10)
    with graph.as_default():
        ((fw_outputs, bw_outputs), (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw=cell,
            inputs=input_tensor,
            sequence_length=[10] * 32,
            dtype=tf.float32,
            swap_memory=True,
            scope=None)
        outputs = tf.concat((fw_outputs, bw_outputs), 2)
        mean = tf.reduce_mean(outputs, axis=1)
        dense = tf.layers.dense(mean, 5, activation=None)

        return dense


def get_opt_op(graph, logits, labels_tensor):
    """Create optimization operation from model's logits and labels

    Args:
        graph (tf.Graph): Tensors' graph
        logits (tf.Tensor): The model's output without activation
        labels_tensor (tf.Tensor): Target labels

    Returns:
        tf.Operation: the operation performing a stem of Adam optimizer
    """
    with graph.as_default():
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels_tensor, name='xent'),
                    name="mean-xent"
                    )
        with tf.variable_scope('optimizer'):
            opt_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
        return opt_op


if __name__ == '__main__':
    # Set random seed for reproducibility
    # and create synthetic data
    np.random.seed(0)
    features = np.random.randn(64, 10, 30)
    labels = np.eye(5)[np.random.randint(0, 5, (64,))]

    graph1 = tf.Graph()
    with graph1.as_default():
        # Random seed for reproducibility
        tf.set_random_seed(0)
        # Placeholders
        batch_size_ph = tf.placeholder(tf.int64, name='batch_size_ph')
        features_data_ph = tf.placeholder(tf.float32, [None, None, 30], 'features_data_ph')
        labels_data_ph = tf.placeholder(tf.int32, [None, 5], 'labels_data_ph')
        # Dataset
        dataset = tf.data.Dataset.from_tensor_slices((features_data_ph, labels_data_ph))
        dataset = dataset.batch(batch_size_ph)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')
        input_tensor, labels_tensor = iterator.get_next()

        # Model
        logits = model(graph1, input_tensor)
        # Optimization
        opt_op = get_opt_op(graph1, logits, labels_tensor)

        with tf.Session(graph=graph1) as sess:
            # Initialize variables
            tf.global_variables_initializer().run(session=sess)
            for epoch in range(3):
                batch = 0
                # Initialize dataset (could feed epochs in Dataset.repeat(epochs))
                sess.run(
                    dataset_init_op,
                    feed_dict={
                        features_data_ph: features,
                        labels_data_ph: labels,
                        batch_size_ph: 32
                    })
                values = []
                while True:
                    try:
                        if epoch < 2:
                            # Training
                            _, value = sess.run([opt_op, logits])
                            print('Epoch {}, batch {} | Sample value: {}'.format(epoch, batch, value[0]))
                            batch += 1
                        else:
                            # Final inference
                            values.append(sess.run(logits))
                            print('Epoch {}, batch {} | Final inference | Sample value: {}'.format(epoch, batch, values[-1][0]))
                            batch += 1
                    except tf.errors.OutOfRangeError:
                        break
            # Save model state
            print('\nSaving...')
            cwd = os.getcwd()
            path = os.path.join(cwd, 'simple')
            shutil.rmtree(path, ignore_errors=True)
            inputs_dict = {
                "batch_size_ph": batch_size_ph,
                "features_data_ph": features_data_ph,
                "labels_data_ph": labels_data_ph
            }
            outputs_dict = {
                "logits": logits
            }
            tf.saved_model.simple_save(
                sess, path, inputs_dict, outputs_dict
            )
            print('Ok')
    # Restoring
    graph2 = tf.Graph()
    with graph2.as_default():
        with tf.Session(graph=graph2) as sess:
            # Restore saved values
            print('\nRestoring...')
            tf.saved_model.loader.load(
                sess,
                [tag_constants.SERVING],
                path
            )
            print('Ok')
            # Get restored placeholders
            labels_data_ph = graph2.get_tensor_by_name('labels_data_ph:0')
            features_data_ph = graph2.get_tensor_by_name('features_data_ph:0')
            batch_size_ph = graph2.get_tensor_by_name('batch_size_ph:0')
            # Get restored model output
            restored_logits = graph2.get_tensor_by_name('dense/BiasAdd:0')
            # Get dataset initializing operation
            dataset_init_op = graph2.get_operation_by_name('dataset_init')

            # Initialize restored dataset
            sess.run(
                dataset_init_op,
                feed_dict={
                    features_data_ph: features,
                    labels_data_ph: labels,
                    batch_size_ph: 32
                }

            )
            # Compute inference for both batches in dataset
            restored_values = []
            for i in range(2):
                restored_values.append(sess.run(restored_logits))
                print('Restored values: ', restored_values[i][0])

    # Check if original inference and restored inference are equal
    valid = all((v == rv).all() for v, rv in zip(values, restored_values))
    print('\nInferences match: ', valid)
```

This will print:

```py
$ python3 save_and_restore.py

Epoch 0, batch 0 | Sample value: [-0.13851789 -0.3087595   0.12804556  0.20013677 -0.08229901]
Epoch 0, batch 1 | Sample value: [-0.00555491 -0.04339041 -0.05111827 -0.2480045  -0.00107776]
Epoch 1, batch 0 | Sample value: [-0.19321944 -0.2104792  -0.00602257  0.07465433  0.11674127]
Epoch 1, batch 1 | Sample value: [-0.05275984  0.05981954 -0.15913513 -0.3244143   0.10673307]
Epoch 2, batch 0 | Final inference | Sample value: [-0.26331693 -0.13013336 -0.12553    -0.04276478  0.2933622 ]
Epoch 2, batch 1 | Final inference | Sample value: [-0.07730117  0.11119192 -0.20817074 -0.35660955  0.16990358]

Saving...
INFO:tensorflow:Assets added to graph.
INFO:tensorflow:No assets to write.
INFO:tensorflow:SavedModel written to: b'/some/path/simple/saved_model.pb'
Ok

Restoring...
INFO:tensorflow:Restoring parameters from b'/some/path/simple/variables/variables'
Ok
Restored values:  [-0.26331693 -0.13013336 -0.12553    -0.04276478  0.2933622 ]
Restored values:  [-0.07730117  0.11119192 -0.20817074 -0.35660955  0.16990358]

Inferences match:  True
```