import click
import sys

from fileutils import read_obj

@click.command()
@click.argument('filename')
def print_stats(filename):
	input = read_obj(filename)
	if input is None:
		print("The specified file does not exist.")
		sys.exit(1)

	sequence_length = int(filename.split("/")[-1].split("_")[0])
	print("Max Rank,Count")

	for i in range(1, 11):
		print("{},{}".format(i, input[str(i)]))

	print("Total,{}".format(input["total"]))
	print("Sequence Length,{}".format(sequence_length))

if __name__ == "__main__":
	print_stats()
