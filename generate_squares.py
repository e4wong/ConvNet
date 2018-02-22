from PIL import Image, ImageDraw
import numpy
from os import listdir
from os.path import isfile, join
from lib import *
import math
def is_square(points, d):
	if len(points) != 4:
		print("messed up")
		return None

	epsilon = 1
	counts = [0,0,0,0]
	for i in range(0, len(points)):
		for j in range(0, len(points)):
			if i == j:
				continue
			if abs(distance(points[i], points[j]) - d) < epsilon:
				counts[i] += 1
			elif abs(distance(points[i], points[j]) - math.sqrt(2*(d**2))) > epsilon:
				return False


	return counts == [2,2,2,2]


for i in range(0,1000):
	im = Image.new("1", (1000, 1000))
	draw = ImageDraw.Draw(im)
	not_found = True
	points = None
	while not_found:
		point1 = generate_random_point(200,800, 200, 800)
		point2 = generate_random_point(200,800, 200, 800)
		d = distance(point1, point2)
		delta_x = point1[0] - point2[0]
		delta_y = point1[1] - point2[1]
		for x in range(0, 1000):
			for y in range(0, 1000):
				point3 = (x, y)
				point4 = (x + delta_x, y +delta_y)
				if x + delta_x >= 1000 or y + delta_y >= 1000 or x + delta_x < 0 or y + delta_y < 0:
					continue
				if is_square([point1, point2, point3, point4], d):
					not_found = False
					points = [point1, point2, point3, point4]
					break
			if not_found == False:
				break
	draw.polygon(points, fill="white")
	im.save("generated_squares/" + str(i+1) + ".png","PNG")