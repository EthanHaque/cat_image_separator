# Testing code to verify model works

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

model = tf.keras.models.load_model(
    r'models\inceptionResNetV2_optimized_h5\inceptionResNetV2_optimized.h5')

path = r"C:\Users\Ethan\Desktop\images"
dataset = image_dataset_from_directory(path,
                                       shuffle=False,
                                       image_size=(299, 299))

predictions = model.predict(dataset)
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

# paths = np.array(os.listdir(path+"/1"))
# paths = paths[..., None]
# paths.reshape((70, 1))
# values = predictions.numpy().reshape((70, 1))

# combined = np.hstack((paths, values))

# print(combined)

print(predictions)
