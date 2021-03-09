#!/usr/bin/env python

import argparse
import os

from DirectoryManipulator import DirectoryManipulator
from Predictor import Predictor


def load(positive_class, negative_class):
    """
    Helper function that loads in the model and creates a DirectoryManipulator.

    :return: The model and a DirectoryManipulator.
    """
    print("#" * 15)
    print("loading model...")
    print("#" * 15)
    p = Predictor(r"models\inceptionResNetV2_optimized_h5\inceptionResNetV2_optimized.h5",
                  (positive_class, negative_class))
    print("#" * 15)
    print("model loaded.")
    print("#" * 15)
    d = DirectoryManipulator()
    return p, d


def create_parser():
    """
    Creates argument parser

    :return: The command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Find images of cats and sort them')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-d', '--directory', type=is_directory, help="performs predictions on entire directory")
    group.add_argument('-dr', '--directory_recursive', type=is_directory,
                       help="performs predictions on entire directory and all sub-directories")
    group.add_argument('-i', '--image', type=is_filepath, help="performs predictions on a single image")

    parser.add_argument('-r', '--reorganize', action='store_true',
                        help="rearranges the file structure of a folder in-place into the given class names")

    parser.add_argument('-p', '--positive', help="name of class for positive classifications", default="cat")
    parser.add_argument('-n', '--negative', help="name of class for negative classifications", default="not_cat")

    parser.add_argument('-w', '--write', help="file path to write output to")

    parser.add_argument('-v', '--verbose', action='store_true', help="displays verbose output")

    return parser.parse_args()


def is_directory(path):
    """
    Checks if path is directory and throws an error if it is not.

    :param path: string of path to a directory.
    :return: the string containing the path to the directory.
    """
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError("the path:\n\n\t{}\n\nis not a valid path to a directory".format(path))

    return path


def is_filepath(path):
    """
    Checks if path is directory and throws an error if it is not.

    :param path: string of path to a directory.
    :return: the string containing the path to the directory.
    """
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError("the path:\n\n\t{}\n\nis not a valid path to a file".format(path))

    return path


def predict_on_image(predictor, path):
    """
    Makes a prediction on a single image.

    :param predictor: the Predictor object to use.
    :param path: string of the path to the image file.
    :return: a string containing either the positive or negative class.
    """
    return predictor.predict_on_image_by_path(path)


def predict_on_directory(predictor, path):
    """
    Makes a prediction on an entire directory of images.

    :param predictor: the Predictor object to use.
    :param path: string of the path to the directory.
    :return: a dictionary with the key being the path to the directory and the value
    being a list of tuples containing the file name and the prediction for that file.
    """
    return predictor.process_directory(path)


def format_predictor_directory_output(input_dict):
    """
    Formats printing the dictionary output from the predictor object.

    :param input_dict: a dictionary with the key being the path to the directory and the value
    being a list of tuples containing the file name and the prediction for that file.
    :return: a string containing the formatted dictionary.
    """
    out = ""
    for keys, values in input_dict.items():
        for tup in values:
            out += "{} ---> {}".format(os.path.join(keys, tup[0]), tup[1].decode("utf8")) + "\n"
        out += "#" * 15 + "\n"

    return out


def main():
    """
    Main functionality of the program
    """

    args = create_parser()
    if args.reorganize:
        if not (args.directory or args.directory_recursive):
            raise argparse.ArgumentTypeError(
                "argument -r/--reorganize is only allowed with -d/--directory and -dr/--directory_recursive")

    p, d = load(args.positive, args.negative)

    prediction = None
    stringify = None

    if args.image:
        prediction = predict_on_image(p, args.image)
        prediction = prediction.decode("utf8")
        stringify = "{} ---> {}".format(args.image, prediction)
    if args.directory:
        prediction = predict_on_directory(p, args.directory)
        stringify = format_predictor_directory_output(prediction)
        if args.reorganize:
            for key, value in list(prediction.items()):
                d.sort_directory(key, value)
    if args.directory_recursive:
        dirs = d.recursively_get_sub_dirs(args.directory_recursive, (args.positive, args.negative))
        prediction = p.process_multiple_directories(dirs)
        stringify = format_predictor_directory_output(prediction)
        if args.reorganize:
            for key, value in list(prediction.items()):
                d.sort_directory(key, value)
    if args.verbose:
        print(stringify)
    if args.write:
        with open(args.write, "w") as file:
            file.write(stringify)


if __name__ == '__main__':
    main()
