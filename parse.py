from PIL import Image
import numpy

directories = ['circle', 'square', 'star', 'triangle']
image = Image.open('circle/998.png')
image = image.resize((200, 200))
image.show()
pix = numpy.array(image)

(width, height) = pix.shape
print (pix[100][100])