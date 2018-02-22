from PIL import Image, ImageDraw
import numpy
from os import listdir
from os.path import isfile, join
from lib import *

for i in range(0,1000):
	im = Image.new("1", (1000, 1000))
	draw = ImageDraw.Draw(im)
	point1 = generate_random_point(200,800, 200, 800)
	point2 = generate_random_point(200,800, 200, 800)
	point3 = generate_random_point(200,800, 200, 800)
	draw.polygon([point1, point2, point3], fill="white")
	im.save("generated_triangles/" + str(i+1) + ".png","PNG")