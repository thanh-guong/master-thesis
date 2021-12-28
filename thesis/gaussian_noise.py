import numpy
from PIL import Image

standard_deviation = 30

IMAGES_DIR = "images/"
INPUT_FILENAME = IMAGES_DIR + "original_lulu_liebe_calicchio.jpg"
OUTPUT_FILENAME = IMAGES_DIR + "noised_sigma" + str(standard_deviation) + "_lulu_liebe_calicchio.jpg"

input_image = Image.open(INPUT_FILENAME)
input_image = numpy.asarray(input_image)

print(input_image[0])

gaussian_noise = numpy.random.normal(0, standard_deviation, (input_image.shape))
input_image = input_image + gaussian_noise

print(input_image[0])

output_image = Image.fromarray(numpy.uint8(input_image))
output_image.save(OUTPUT_FILENAME, "JPEG")
