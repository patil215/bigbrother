import click
import sys
import os
import copy

from fileutils import read_obj, write_obj, get_next_file_number
from tracepoint import TracePath

@click.command()
@click.argument("longpath")
@click.option("-p", "--padding", type=click.INT, required=False, default=0,
    help="""Pad the beginning and end of each digit sequence by
    a number of frames. Defaults to zero (0).
    """)
def generate_subpaths(longpath, padding):
    if not os.path.exists(longpath):
        print("Invalid long path provided.")
        sys.exit(1)

    path = read_obj(longpath)
    checkpoint_indices = sorted(list(path.checkpoint_indices))

    base_folder_name = longpath[0:os.path.dirname(longpath).rfind('/')]
    path_folder_name = base_folder_name.split('/')[-1]
    base_folder_name = base_folder_name[0:base_folder_name.rfind('/')]
    base_folder_name = base_folder_name[0:base_folder_name.rfind('/')]

    class_names = path_folder_name.split('_')
    num_classes = len(class_names)

    print("class names: {}".format(class_names))
    print("number of classes: {}".format(num_classes))
    print("path checkpoints: {}".format(checkpoint_indices))

    for subpath_digit_length in range(1, num_classes):
        left_digit_class_index = 0
        right_digit_class_index = left_digit_class_index + subpath_digit_length

        for left_digit_checkpoint_index in range(0, len(checkpoint_indices) - (subpath_digit_length * 2) + 1, 2):
            right_digit_checkpoint_index = left_digit_checkpoint_index + (subpath_digit_length * 2) - 1

            # compute path indices including padding
            left_frame_index = max(0, checkpoint_indices[left_digit_checkpoint_index] - padding)
            right_frame_index = min(len(path.path) - 1, checkpoint_indices[right_digit_checkpoint_index] + padding)
            
            # get checkpoints included in path indices
            middle_checkpoints = checkpoint_indices[left_digit_checkpoint_index:right_digit_checkpoint_index + 1]
            
            # realign checkpoint locations to new indices
            adjusted_checkpoints = [checkpoint - left_frame_index for checkpoint in middle_checkpoints]
            print("subpath length: {}, left index: {}, right index: {}, middle_checkpoints: {}, adjusted_checkpoints: {}"
                .format(subpath_digit_length, left_frame_index, right_frame_index, middle_checkpoints, adjusted_checkpoints))

            included_class_names = class_names[left_digit_class_index:right_digit_class_index]
            print("classes included: {}".format(included_class_names))
            output_file_folder = "{}/{}_digits/{}/paths".format(base_folder_name, len(included_class_names), "_".join(included_class_names))
            file_number = get_next_file_number(output_file_folder)
            output_file_name = "{}/{}.path".format(output_file_folder, file_number)

            # extract subpath and realign timestamps
            subpath = copy.deepcopy(path.path[left_frame_index:right_frame_index + 1])
            initial_timestamp = subpath[0].t
            for point in subpath:
                point.t = point.t - initial_timestamp

            new_tracepath = TracePath(subpath, set(adjusted_checkpoints))
            write_obj(output_file_name, new_tracepath)

            left_digit_class_index += 1
            right_digit_class_index = left_digit_class_index + subpath_digit_length

if __name__ == '__main__':
    generate_subpaths()