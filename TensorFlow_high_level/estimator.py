import os

import six.moves.urllib.request as request
import tensorflow as tf

# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
print("TensorFlow version: {}".format(tf_version))
# assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"

# Windows users: You only need to change PATH, rest is platform independent
PATH = "tf_dataset_and_estimator_apis"

# Fetch and store Training and Test dataset files
PATH_DATASET = PATH + os.sep + "dataset"
FILE_TRAIN = PATH_DATASET + os.sep + "iris_training.csv"
FILE_TEST = PATH_DATASET + os.sep + "iris_test.csv"
URL_TRAIN = "http://download.tensorflow.org/data/iris_training.csv"
URL_TEST = "http://download.tensorflow.org/data/iris_test.csv"


def download_dataset(url, file):
    if not os.path.exists(PATH_DATASET):
        os.makedirs(PATH_DATASET)
    if not os.path.exists(file):
        data = request.urlopen(url).read()
        with open(file, "wb") as f:
            f.write(data)
            f.close()


download_dataset(URL_TRAIN, FILE_TRAIN)
download_dataset(URL_TEST, FILE_TEST)

tf.logging.set_verbosity(tf.logging.INFO)

# The CSV features in our training & test data
feature_names = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth']


# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API

def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1]  # Last element is the label
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything but last elements are the features
        d = dict(zip(feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path)  # Read text file 读取文本
               .skip(1)  # Skip header row # 跳过第一行表头
               .map(decode_csv))  # Transform each elem by applying decode_csv fn # 将数据进行转换
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(32)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


next_batch = my_input_fn(FILE_TRAIN, True)  # Will return 32 random elements

# Now let's try it out, retrieving and printing one batch of data.
# Although this code looks strange, you don't need to understand
# the details.
with tf.Session() as sess:
    first_batch = sess.run(next_batch)
print(first_batch)

feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

# Create a deep neural network regression classifier
# Use the DNNClassifier pre-made estimator
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,  # The input features to our model
    hidden_units=[10, 10],  # Two layers, each with 10 neurons
    n_classes=3,
    model_dir=PATH)  # Path to where checkpoints etc are store

classifier.train(
    input_fn=lambda: my_input_fn(FILE_TRAIN, True, 8))

evaluate_result = classifier.evaluate(
    input_fn=lambda: my_input_fn(FILE_TEST, False, 4))
print("Evaluation results")
for key in evaluate_result:
    print("   {}, was: {}".format(key, evaluate_result[key]))

predict_results = classifier.predict(
    input_fn=lambda: my_input_fn(FILE_TEST, False, 1))
print("Predictions on test file")
for prediction in predict_results: # 预测结果
   # Will print the predicted class, i.e: 0, 1, or 2 if the prediction
   # is Iris Sentosa, Vericolor, Virginica, respectively.
   print (prediction["class_ids"][0])