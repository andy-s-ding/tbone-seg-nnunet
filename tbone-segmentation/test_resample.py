"""test_resample.py
"""

import ants
import nrrd


def main():

	test = ants.image_read("Users/alex/Documents/predictions/RT_153_epsilon-downsample300-syn80-demons.seg.nrrd")
	final = nrrd.read_header("/Users/alex/Documents//Users/alex/Downloads/OneDrive_1_1-14-2021/EPSILON/CBCT_UNEMBEDDED_rotated.nrrd")

	
	return


if __name__ == "__main__":

	main()