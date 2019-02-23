import pandas as pd
import numpy as np
import os

# person class def
class Person(object):

	def __init__(self, data):
		self.data = data

	def get_data(self):
		return self.data

	def get_name(self):
		return self.data[0]

	def get_sex(self):
		return self.data[1]

	def get_dob(self):
		return self.data[2]

	def get_education(self):
		return self.data[3]

	def get_address(self):
		return self.data[4]

	def get_phone(self):
		return self.data[5]

	def get_email(self):
		return self.data[6]

	def anonymise(self, data):
		''' 
			data: a list of values
		'''
		# TODO

def load_csv(path=None, col_names=None):
	''' loads a csv file into a panda dataframe '''
	res = None
	if path is None:
		fname = 'people_test_data_csv.csv'
		path = os.path.join(os.getcwd(), fname)
	try:
		res = pd.read_csv(path, names=col_names) if col_names else pd.read_csv(path)
	except FileNotFoundError:
		print("{} does not exist".format(path))
	return res

if __name__ == '__main__':
	
	df = load_csv() # load the data

	# print the whole data
	# print(df)

	# print data from specific columns, e.g. name and phone
	# print(df[['Name', 'Phone']])
	
	# loops through each row and creates a person object, adding the results to a list of people
	# people = []
	# for row in df.values:
	# 	p = Person(data=row)
	# 	people.append(p)

	

		
