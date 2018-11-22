import click
import sys
import os

from fileutils import read_obj, write_obj

@click.command()
@click.argument("longpath")
def subpath(longpath):
    if not os.path.exists(longpath):
        print("Invalid long path provided.")
        sys.exit(1)

    path = read_obj(longpath)
    checkpoint_indices = sorted(list(path.checkpoint_indices))

    path_folder_name = os.path.dirname(longpath).split('/')[-2]
    class_names = path_folder_name.split('_')
    num_classes = len(class_names)

    print("class names: {}".format(class_names))
    print("number of classes: {}".format(num_classes))
    print("path checkpoints: {}".format(checkpoint_indices))

    for subpath_length in range(1, num_classes):
        left_digit_class_index = 0
        right_digit_class_index = left_digit_class_index + subpath_length

        for left_digit_checkpoint_index in range(0, len(checkpoint_indices) - (subpath_length * 2) + 1, 2):
            right_digit_checkpoint_index = left_digit_checkpoint_index + (subpath_length * 2) - 1

            left_frame_index = checkpoint_indices[left_digit_checkpoint_index]
            right_frame_index = checkpoint_indices[right_digit_checkpoint_index]
            middle_checkpoints = checkpoint_indices[left_digit_checkpoint_index:right_digit_checkpoint_index + 1]
            print("subpath length: {}, left: {}, right: {}, checkpoints: {}".format(subpath_length, left_frame_index, right_frame_index, middle_checkpoints))

            print("classes included: {}".format(class_names[left_digit_class_index:right_digit_class_index]))
            left_digit_class_index += 1
            right_digit_class_index = left_digit_class_index + subpath_length

if __name__ == '__main__':
    subpath()