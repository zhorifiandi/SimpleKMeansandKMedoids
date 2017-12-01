import numpy

# Input must be tuple of integer / float
def countDistance(centroid, instance):
	numpy_centroid = numpy.array(centroid)
	numpy_instance = numpy.array(instance)
	return numpy.linalg.norm(numpy_centroid-numpy_instance)

def create_list_dataset(filename):
	file = open(filename, 'r')
	list_output = []
	for line in file:
		list_temp = []
		for attr in line.split(",")[:-1]:
			attr = attr.strip()
			if attr.isdigit():
				list_temp.append(int(attr))
		if len(list_temp) > 0:
			list_output.append(list_temp)
	list_output = list_output
	return list_output
