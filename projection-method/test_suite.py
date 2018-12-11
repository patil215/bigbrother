import click
from fileutils import get_test_angle_paths, write_obj
from project import estimate_paper_rotation
from predict_batch import test_batch
from collections import defaultdict
import easygui

@click.command()
@click.argument("test_dir")
@click.option("-d", "--data", help="Location of the data directory", default="data")
@click.option("-o", "--output", help="Directory to dump output data", default="output")
def test_suite(test_dir, data, output):
	angle_paths = get_test_angle_paths(test_dir)

	# Create frames that autodetect the angle
	print("Determining angles for frames...")
	angles = {}
	for path in angle_paths:
		raw = easygui.enterbox("Please specify angle for {} (3 integers separated by space).".format(path))
		angles[path] = tuple(map(int, raw.split(" ")))

	# Iterate through all possible sequence lengths and take data for them
	for length in range(3, 10): # Exclude 10 digit
		all_path_statistics = []
		for path in angle_paths:
			statistics, classifications = test_batch("{}/{}_digits/".format(path, length), data, angles[path], length)

			# Write the file
			specific_output_path = "{}/{}/{}_digits".format(output, path.split("/")[-1], length)
			print("Saving statistics for angle {} and length {}".format(path, length))
			print(specific_output_path)
			print(statistics)
			print(dict(classifications))
			write_obj("{}.statistics".format(specific_output_path), statistics)
			write_obj("{}.classifications".format(specific_output_path), dict(classifications))

			# We need this to compute statistics across all paths
			all_path_statistics.append(dict(statistics))

		print("Saving global statistics...")
		all_statistics = defaultdict(int)
		for stats in all_path_statistics:
			for prop in stats:
				all_statistics[prop] += stats[prop]
		print(dict(all_statistics))
		write_obj("{}/{}_digits.statistics".format(output, length), dict(all_statistics))

if __name__ == "__main__":
	test_suite()