import click
import sys
import os

from fileutils import read_obj

def write_csv(filename):
	print(filename)
	input = read_obj(filename)
	if input is None:
		print("The specified file does not exist.")
		sys.exit(1)

	sequence_length = int(filename.split("/")[-1].split("_")[0])
	output_filename = "{}{}_digits.csv".format(filename[:filename.rfind('/') + 1], sequence_length)
	output_file = open(output_filename, 'w')
	print("Max Rank,Count", file=output_file)

	for i in range(1, 11):
		print("{},{}".format(i, input[str(i)]), file=output_file)

	print("Total,{}".format(input["total"]), file=output_file)
	print("Sequence Length,{}".format(sequence_length), file=output_file)

@click.command()
@click.argument('folder')
def print_stats(folder):
	subfiles = ["{}/{}".format(folder, f.name) for f in os.scandir(folder)
		if not f.is_dir() and "statistics" in f.name]

	for stats_file in subfiles:
		write_csv(stats_file)

if __name__ == "__main__":
	print_stats()
