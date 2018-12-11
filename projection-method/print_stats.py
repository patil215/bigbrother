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

	print(input)

if __name__ == "__main__":
	print_stats()
