import numpy
from sklearn.preprocessing import normalize


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

def normalize_attr(list_dataset):
	print("NORMALIZING... \nPLEASE WAIT...")
	array_to_normalize = numpy.array(list_dataset)
	array_to_normalize = normalize(array_to_normalize)
	list_intance = []
	for instance in array_to_normalize:
		list_attr = []
		for attr in instance:
			list_attr.append(attr)
		list_intance.append(list_attr)
	return list_intance

def create_list_label(filename):
	file = open(filename, 'r')
	list_dataset = []
	for line in file:
		if len(line) > 2:
			list_dataset.append(line.split(","))
	# Asumption label value located at last element of list
	
	# Create Map
	b = 0
	dict_label = {}
	for instance in list_dataset:
		if str(instance[-1]) not in dict_label:
			dict_label[str(instance[-1])] = b
			b += 1

	# Create list label
	list_label = []
	for instance in list_dataset:
		list_label.append(dict_label[str(instance[-1])])

	return list_label