import os
import shutil


class DirectoryManipulator:
    """This class reorganizes files"""

    @staticmethod
    def recursively_get_sub_dirs(root_dir, exclude):
        """
        Recursively searches a directory for child directories.

        :param exclude: does not traverse these directories
        :param root_dir: The path to the root directory to search.
        :return: list containing all the given directory and all its child directories.
        """

        out_dirs = []
        for root, directories, files in os.walk(root_dir, topdown=True):
            directories[:] = [dpth for dpth in directories if dpth not in exclude]
            for dpth in directories:
                out_dirs.append(os.path.join(root, dpth))
        return out_dirs

    @staticmethod
    def sort_directory(directory, sub_dirs):
        """
        Takes a directory and a list of tuples containing a file name and new directory name to place a file into.
        :param directory: the path to the root directory.
        :param sub_dirs: list of tuples containing a file name and a new directory name to place that file into.
        """
        for item in sub_dirs:
            current_path = os.path.join(directory, item[0])
            destination = os.path.join(directory, item[1].decode("utf8"))
            if not os.path.exists(destination):
                os.makedirs(destination)
            shutil.move(current_path, destination)


if __name__ == "__main__":
    # from Predictor import Predictor
    #
    # class_names = ("cat", "not_cat")
    # d = DirectoryManipulator()
    # p = Predictor(r"models\inceptionResNetV2_optimized_h5\inceptionResNetV2_optimized.h5", class_names)
    # path = r"C:\Users\Ethan\Desktop\cat_image_separator\data\testing"
    # dirs = d.recursively_get_sub_dirs(path, class_names)
    #
    # predictions = p.process_directory(dirs[2])
    # predictions = list(predictions.values())[0]
    # d.sort_directory(dirs[2], predictions)
    pass
