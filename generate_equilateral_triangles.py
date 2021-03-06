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
	while True:
		im = Image.new("1", (1000, 1000))
		draw = ImageDraw.Draw(im)
		point1 = generate_random_point(200,800, 200, 800)
		point2 = generate_random_point(200,800, 200, 800)
		d = distance(point1, point2)
		min_d = d * 10
		min_point = None
		for x in range(0, 1000):
			for y in range(0, 1000):
				point = (x, y)
				d1 = distance(point, point1)
				d2 = distance(point, point2)
				points = [point1, point2, point]

				if abs(d1 - d) < min_d and abs(d2 - d) < min_d:
					min_point = point
					min_d = max(abs(d1 - d), abs(d2 - d))
		if too_small([point1, point2, min_point]):
			continue
		draw.polygon([point1, point2, min_point], fill="white")
		im.save("generated_equilateral_triangles/" + str(i+1) + ".png","PNG")
		break