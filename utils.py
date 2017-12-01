import numpy

# Input must be tuple of integer / float
def countDistance(centroid, instance):
	numpy_centroid = numpy.array(centroid)
	numpy_instance = numpy.array(instance)
	return numpy.linalg.norm(numpy_centroid-numpy_instance)