from PIL import Image, ImageDraw
import numpy
from os import listdir
from os.path import isfile, join
from lib import *
import math

def too_small(points):
	if abs(points[0][0] - points[1][0]) < 300:
		return True
	if abs(points[0][1] - points[1][1]) < 300:
		return True
	return False


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
		point1 = generate_random_point(100,900, 100, 900)
		x_search = list(range(point1[0], 900))
		random.shuffle(x_search)
		y_search = list(range(point1[1], 900))
		random.shuffle(y_search)
		for x in x_search:
			for y in y_search:
				width = x - point1[0]
				height = y - point1[1]
				if abs(width - height) < 1:
					new_point = (x,y)
					points = [point1, new_point]
					if too_small(points):
						continue
					else:
						not_found = False
						break
			if not(not_found):
				break

	draw.ellipse(points, fill="white")
	im.save("generated_circles/" + str(i+1) + ".png","PNG")