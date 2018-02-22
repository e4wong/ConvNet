from PIL import Image, ImageDraw
import numpy
from os import listdir
from os.path import isfile, join
from lib import *

def too_small(points):
	for i in range(0, len(points)):
		distances = []
		for j in range(0, len(points)):
			if i == j:
				continue
			distances.append(distance(points[i], points[j]))
		for d in distances:
			if d < 300:
				return True
	return False

for i in range(0,1000):
	points = None
	while True:
		im = Image.new("1", (1000, 1000))
		draw = ImageDraw.Draw(im)
		point1 = generate_random_point(200,800, 200, 800)
		point2 = generate_random_point(200,800, 200, 800)
		point3 = generate_random_point(200,800, 200, 800)
		points = [point1, point2, point3]
		if !too_small(points):
			break
	draw.polygon(points, fill="white")
	im.save("generated_triangles/" + str(i+1) + ".png","PNG")