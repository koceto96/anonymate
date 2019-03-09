import random
import string
import csv
import re
import sys
from ukpostcodeutils import validation

#formula is town^3+10*region
SCORES_BY_AREA =  {
					"W":1, "SW":3, "NW":6, "LS":12, "L":14, "B":14, "LE":15, "SR":18,
 					"S":18, "HU":18, "PL":19, "ST":21, "SA":21, "DY":22, "DH":24
				  }

AGE_RANGE = (18, 73)

# list of all capital letters
CAPITAL_ASCII = string.ascii_letters[26:]

SALARY_RANGE = (11000, 2400000)

RANGE_BY_AREA = { 
					'W' : (2, 4), 'SW' : (2, 4), 'NW' : (1, 3), 'LS' : (1, 3), 'L' : (1, 3),
					'B' : (1, 3), 'LE' : (1, 3), 'SR' : (1, 3), 'S' : (1, 3), 'HU' : (1, 3),
					'PL': (1, 3), 'ST' : (1, 3), 'SA' : (1, 3), 'DY': (1, 3), 'DH' : (6, 8)
			    }

FIRST_NAMES = ['John', 'Jack', 'James', 'Daniel', 'Peter', 'Ana', 'Abbey', 'Mary',
			  'Jake', 'Simon', 'William', 'Vanessa', 'Victoria', 'Nick', 'Diane',
			  'Michael', 'Michelle', 'Sarah', 'Sophie', 'Richard', 'Alexandra',
			  'Spencer', 'Kate', 'Elizabeth', 'Anthony', 'Kieran', 'Tanya']

SURNAMES = ['Williams', 'Johnson', 'Webb', 'Gibbs', 'Cole', 'Ramsey', 'Turner',
			'Wallace', 'Robinson', 'Parker', 'Son', 'Kane', 'Greenford', 'Rashford',
			'Sturridge', 'Sterling', 'Aguero', 'Gomez', 'Suarez', 'Sanchez',
			'Rodriguez', 'Wright', 'Kyle', 'Crawley', 'Smith', 'Connor', 'Jameson']


# generates n random digits
def rand_n_digits(n):
    s = 10**(n - 1)
    e = (10**n) - 1
    return random.randint(s, e)

def generate_valid_postcode():
	part1, part2 = generate_postcode_in_parts()
	while not validation.is_valid_postcode(part1+part2):
		part1, part2 = generate_postcode_in_parts()
	return "{0} {1}".format(part1, part2)

def generate_postcode_in_parts():
	pArea = random.choice(RANGE_BY_AREA.keys())
	pAreaRange = RANGE_BY_AREA.get(pArea, [2, 3])
	pAreaNum = random.randrange(pAreaRange[0], pAreaRange[1] + 1)
	pLastNum = random.randint(0, 9)
	pLastCode = "{0}{1}".format(random.choice(CAPITAL_ASCII), random.choice(CAPITAL_ASCII))
	part1 = "{0}{1}".format(pArea, pAreaNum)
	part2 = "{0}{1}".format(pLastNum, pLastCode)
	return (part1, part2)

def area_pairs():
	# [ 
	#   [('W', 4), ('W', 3), ('W', 2)], [('SW', 4), ('SW', 3), ('SW', 2)], ...,
	#   [('DH', 8), ('DH', 7), ('DH', 6)]
	# ]
	lstPairs = list()
	for area, nr in RANGE_BY_AREA.items():
		nrList = list(range(nr[1], nr[0] - 1, -1))
		lstPairs.append([ (area, n) for n in nrList ])
	return lstPairs

def getKeysByVal(aDict, val):
	return [ k for k, v in aDict.items() if v == val ]

def generate_income(postcode_part1, age):

	def sal_by_slot(num_slots, lowerB, upperB):
		num_slots = sorted(num_slots, reverse=True)
		salaryBySlot = dict()
		s_inc = 10000
		s_lower = lowerB
		for slot in num_slots[:-3]:
			s_upper = s_lower + s_inc
			sRange = (s_lower, s_upper)
			salaryBySlot[slot] = sRange
			s_lower = s_upper / 2
			s_inc += 10000

		s_inc = (upperB - s_lower) / 3
		for slot in num_slots[-3:]:
			s_upper = s_lower + s_inc if slot != num_slots[-1] else upperB
			sRange = (s_lower, s_upper)
			salaryBySlot[slot] = sRange
			s_lower = s_upper / 2

		return salaryBySlot

	num_areas = set(SCORES_BY_AREA.values())
	salaryBySlot = sal_by_slot(num_areas, SALARY_RANGE[0], SALARY_RANGE[1])

	sRangeByArea = dict()
	for slot, sRange in salaryBySlot.items():
		areas = getKeysByVal(SCORES_BY_AREA, slot)
		for area in areas:
			sRangeByArea[area] = sRange

	sRangeBySubArea = dict()
	for lstPairs in area_pairs():
		sRange = sRangeByArea[lstPairs[0][0]]
		lowerB, upperB = sRange
		s_inc = (upperB - lowerB) / 3
		curr_lower_bound = lowerB
		for areaPair in lstPairs:
			curr_upper_bound = curr_lower_bound + s_inc
			curr_s_range = (curr_lower_bound, curr_upper_bound)
			sRangeBySubArea[areaPair] = curr_s_range
			curr_lower_bound = curr_upper_bound

	incomeByAreaAge = dict()
	for k, v in sRangeBySubArea.items():
		s_inc = (v[1] - v[0]) / (AGE_RANGE[1] - AGE_RANGE[0])
		income = v[0]
		for theAge in range(AGE_RANGE[0], AGE_RANGE[1] + 1):	
			areaAge = ("{0}{1}".format(k[0], k[1]), theAge)
			income += s_inc
			incomeByAreaAge[areaAge] = income

	return incomeByAreaAge[(postcode_part1, age)]


def generate_person_details():
	name = "{0} {1}".format(random.choice(FIRST_NAMES), random.choice(SURNAMES))
	postcode = generate_valid_postcode()
	phone = "{0}{1}".format('07', rand_n_digits(9))
	age = random.randrange(AGE_RANGE[0], AGE_RANGE[1] + 1)
	income = generate_income(postcode.split(' ')[0], age)
	return [name, postcode, phone, age, income]

if __name__ == '__main__':
	data = [['name', 'postcode', 'phone', 'age', 'income']]
	people = 50
	if len(sys.argv) > 1:
		people = int(sys.argv[1])
	for i in xrange(people):
		data.append(generate_person_details())

	with open('csv_data.csv', 'w') as outputFile:
		writer = csv.writer(outputFile)
		writer.writerows(data)
	outputFile.close()

	