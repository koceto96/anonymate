# Anonymate
A privacy preserving data extractor written in `Python`.

## Description

A facility that takes a large dataset of people's personal information and anonymises it up to a state, where it is impossible to distinguish between the people. However, it also preserves the data's structure and 'sense', which will allow for meaningful data science experiments. There are a lot of unexplored structure preserving algorithms in Mathematics that will be used for the task at hand, one simple example being the Chinese Remainder Theorem while more advanced studies might involve Group/Category theory. 

## Running the program
In order to test the claim that data analysis isn't impacted by the data anonymization, machine learning will be performed on a file (data.csv), containing randomly generated (via the generate_csv_data.py script) user data and results will be compared.

Ensure data.csv, comparingResults.py and anonymiseData.py are in the same folder. All the needed libraries for the program should also be installed in this folder or visible from this folder.

Then execute the following line:      python comparingResults.py

Machine learning will then be performed with original user data from data.csv and with the anonymized version of the data in that file. The machine learning results in each case will be displayed, proving that the quality of results isn't impacted by the data anonymization. The first few lines of data for each version are also printed out, allowing one to see the way in which data has been anonymized.
