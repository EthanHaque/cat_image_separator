import tensorflow as tf
import os
import numpy as np


class Predictor:
    """This class is used to predict whether an image is of a cat or not."""

    def __init__(self, path_to_model, classes=(1, 0)):
        """
        Constructor for the Predictor class.

        :param path_to_model: string of path to the model file to load in.
        """
        self.model = self.load_model(path_to_model)
        self.extensions = [".jpg", ".jpeg", ".png"]
        self.classes = classes

    @staticmethod
    def read_files(directory, extensions):
        """
        Generates all the paths to files with given extensions in a directory.

        :param directory: string containing the path to a directory.
        :param extensions: list of strings containing the extensions to include.
        :return: list of strings that contain all the paths to the files with the given extensions in a directory.
        """
        paths = [directory + "\\" + f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in extensions]

        return paths

    @staticmethod
    def read_image(path, size=(299, 299)):
        """
        Reads in an image from a given path and resizes it.

        :param path: string containing the path to an image.
        :param size: optional size 2 tuple of ints to resize the image to.
        :return: image object processed and resized.
        """
        image = tf.keras.preprocessing.image.load_img(path, color_mode="rgb")
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.image.resize(image, size)
        image = np.expand_dims(image, axis=0)

        return image

    @staticmethod
    def load_model(path):
        """
        Loads the tensorflow model.

        :param path: string of path to the model.
        :return: the loaded model.
        """
        return tf.keras.models.load_model(path)

    def process_prediction(self, prediction):
        """
        Takes a prediction tensor and processes the value.

        :param prediction: the tensor containing the prediction.
        :return: the prediction value mapped to its corresponding class.
        """
        prediction = tf.nn.sigmoid(prediction)
        prediction = tf.where(prediction < 0.5, self.classes[0], self.classes[1])
        prediction = prediction.numpy()[0][0]

        return prediction

    def process_directory(self, directory):
        """
        Takes in a directory and does a prediction on all image files in the directory.

        :param directory: string of path to directory.
        :return: dictionary where the key is the path to the directory
        and the value is a list of tuples containing the path to the image and the prediction value.
        """
        paths = self.read_files(directory, self.extensions)

        predictions = {directory: []}
        for path in paths:
            image = self.read_image(path)
            prediction = self.predict_on_image(image)
            prediction = (os.path.basename(path), prediction)
            predictions[directory].append(prediction)

        return predictions

    def predict_on_image_by_path(self, path):
        image = self.read_image(path)
        return self.predict_on_image(image)

    def predict_on_image(self, image):
        """
        Takes in an image and predicts what class it belongs to.

        :param image: an image object.
        :return: the prediction value mapped to the corresponding class for an image.
        """
        prediction = self.model.predict(image)
        prediction = self.process_prediction(prediction)

        return prediction

    def test_func(self, directory):
        """
        Test function that processes a directory.

        :param directory: string of path to a directory.
        """
        predictions = self.process_directory(directory)

        print(predictions)

    def process_multiple_directories(self, directories):
        """
        Processes several directories.
        :param directories: list of string of paths to directories
        :return: dictionary where the key is the path to the directory
        and the value is a list of tuples containing the path to the image and the prediction value.
        """
        predictions = {}
        for directory in directories:
            prediction = self.process_directory(directory)
            predictions.update(prediction)

        return predictions


if __name__ == "__main__":
    ptd = [r"C:\Users\Ethan\Desktop\cat_image_separator\data\testing\1",
           r"C:\Users\Ethan\Desktop\cat_image_separator\data\testing\2",
           r"C:\Users\Ethan\Desktop\cat_image_separator\data\testing\3"]
    ptm = r"models\inceptionResNetV2_optimized_h5\inceptionResNetV2_optimized.h5"
    p = Predictor(ptm, ("cat", "not_cat"))
    ptds = p.process_multiple_directories(ptd)
    for key, val in ptds.items():
        print(key, end=":\n")
        for v in val:
            print(v)
