from PIL import Image, ImageDraw
import numpy
from os import listdir
from os.path import isfile, join
from lib import *
import math
import random

def too_small(points):
	for i in range(0, len(points)):
		distances = []
		for j in range(0, len(points)):
			if i == j:
				continue
			distances.append(distance(points[i], points[j]))
		for d in distances:
			if d < 250:
				return True
	return False

def is_rectangle(points, d):
	if len(points) != 4:
		print("messed up")
		return None

	epsilon = 1
	counts = [0,0,0,0]
	for i in range(0, len(points)):
		distances = []
		for j in range(0, len(points)):
			if i == j:
				continue
			distances.append(distance(points[i], points[j]))
			# so it has to be less square to pass the check
			if abs(distance(points[i], points[j]) - d) < 50 * epsilon:
				counts[i] += 1
		distances.sort()
		a2 = distances[0] ** 2
		b2 = distances[1] ** 2
		c2 = distances[2] ** 2
		if abs(a2 + b2 - c2) > 1:
			return False

	print(counts)
	if counts == [2,2,2,2]:
		print ("was basically a square")
		if random.uniform(0,1) < .80:
			print ("was too square but got unlucky")
			return False
	return True


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
		x_search = list(range(0, 1000))
		random.shuffle(x_search)
		y_search = list(range(0, 1000))
		random.shuffle(y_search)
		for x in x_search:
			for y in y_search:
				point3 = (x, y)
				point4 = (x + delta_x, y +delta_y)
				if x + delta_x >= 1000 or y + delta_y >= 1000 or x + delta_x < 0 or y + delta_y < 0:
					continue
				if is_rectangle([point1, point2, point3, point4], d):
					points = [point1, point2, point3, point4]
					if too_small(points):
						continue
					not_found = False
					break
			if not_found == False:
				break
	draw.polygon(points, fill="white")
	im.save("generated_rectangles/" + str(i+1) + ".png","PNG")