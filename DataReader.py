from tensorflow.keras.preprocessing import image_dataset_from_directory


class DataReader:
    """ Manipulates and reads in data. """

    def __init__(self, train_path, validation_path, batch_size, image_size):
        """
        Reads in different paths to a data.Dataset

        :param train_path: string of the path to the training dataset
        :param validation_path: string of the path toe the validation dataset
        :param batch_size: int containing the batch size to use
        :param image_size: tuple of size 2 containing the size to resize the images to
        """
        self.train_dataset = image_dataset_from_directory(train_path,
                                                          shuffle=True,
                                                          batch_size=batch_size,
                                                          image_size=image_size)

        self.validation_dataset = image_dataset_from_directory(validation_path,
                                                               shuffle=True,
                                                               batch_size=batch_size,
                                                               image_size=image_size)


if __name__ == "__main__":
    tp = r"C:\Users\Ethan\Desktop\filter_cat_pictures\data\train"
    vp = r"C:\Users\Ethan\Desktop\filter_cat_pictures\data\validation"
    bs = 32
    sz = (299, 299)

    dr = DataReader(tp, vp, bs, sz)

    print(dr.validation_dataset)
