def add_dict(dict1, dict2):
	# Need to copy dictionaries to prevent changing der for original variables
	dict1 = dict(dict1)
	dict2 = dict(dict2)
	for key in dict2:
		if key in dict1:
			dict2[key] = dict2[key] + dict1[key]
	return {**dict1, **dict2}