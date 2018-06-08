import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.enable_eager_execution()

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

# Reshape from (N, 28, 28) to (N, 784)
train_images = np.reshape(train_images, (TRAINING_SIZE, 784))
test_images = np.reshape(test_images, (TEST_SIZE, 784))

# Convert the array to float32 as opposed to uint8
train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)

# Convert the pixel values from integers between 0 and 255 to floats between 0 and 1
train_images /= 255
test_images /= 255

NUM_DIGITS = 10

print("Before", train_labels[0])  # The format of the labels before conversion

train_labels = tf.keras.utils.to_categorical(train_labels, NUM_DIGITS)

print("After", train_labels[0])  # The format of the labels after conversion

test_labels = tf.keras.utils.to_categorical(test_labels, NUM_DIGITS)
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)
BATCH_SIZE = 128

# Because tf.data may work with potentially **large** collections of data
# we do not shuffle the entire dataset by default
# Instead, we maintain a buffer of SHUFFLE_SIZE elements
# and sample from there.
SHUFFLE_SIZE = 10000

# Create the dataset
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset = dataset.shuffle(SHUFFLE_SIZE)
dataset = dataset.batch(BATCH_SIZE)

for image, label in dataset:
    print(image)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Create a TensorFlow optimizer, rather than using the Keras version
# This is currently necessary when working in eager mode
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

# We will now compile and print out a summary of our model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()
EPOCHS = 31

x_epochs = range(EPOCHS)
y_loss = []
y_accuracy = []

for epoch in range(EPOCHS):
    for images, labels in dataset:
        train_loss, train_accuracy = model.train_on_batch(images, labels)
    y_loss += [train_loss]
    y_accuracy += [train_accuracy]
    if epoch > 0 and epoch % 5 == 0:
        print('Epoch #%d\t Loss: %.6f\tAccuracy: %.6f' % (epoch, train_loss, train_accuracy))

# plt.subplot(221)
plt.plot(x_epochs, y_loss)
plt.title("Learning curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# plt.subplot(222)
plt.plot(x_epochs, y_accuracy)
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

