from PIL import Image
import numpy
from os import listdir
from os.path import isfile, join

directories = ['circle', 'square', 'star', 'triangle']
output_file = "shape_dataset"
of = open(output_file, 'w')
for directory in directories:
	files = [f for f in listdir(directory) if isfile(join(directory, f))]
	for file in files:
		image = Image.open(directory + '/' + file)
		image = image.resize((50, 50))
		pix = numpy.array(image)
		(width, height) = pix.shape
		label = directories.index(directory)
		s = str(label) + " "
		for i in range(0, width):
			for j in range(0, height):
				s += str(pix[i][j]/255.0) + ","
			s = s[:len(s) - 1]
			s += "?"
		s = s[:len(s) - 1]
		s = s.replace('\n', '')
		s += "\n"
		of.write(s)