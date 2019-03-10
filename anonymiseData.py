import pandas as pd

def putStars (length):
    res = ""
    for i in range (length):
        res += "*"
    return res        

def anonimyse(fileName):
    file = pd.read_csv(fileName, encoding = "ISO-8859-1")
    i=0
    for entry in file ["name"]:
        file.at[i, 'name'] = putStars(len(str(entry)))
        i += 1
    i=0
    #to convert the fields to int
    file ["tel"] = file ["tel"].astype('U')
    for entry in file ["tel"]:
        file.at[i, 'tel'] = putStars(len(str(entry)))
        i += 1    
    i=0
    for entry in file ["postcode"]:
        #replace 2nd half with *'s as not needed for ML
        file.at[i, 'postcode'] = str(entry.split()[0]) + putStars(len(entry.split()[1]))
        i += 1
    i=0
    file ["age"] = file ["age"].astype('U')
    for entry in file ["age"]:
        age = int(entry)
        if 17 < age <= 23:
            file.at[i, 'age'] = '18-23'
        elif 23 < age <= 33:
            file.at[i, 'age'] = '23-33'
        elif 33 < age <= 43:
            file.at[i, 'age'] = '33-43'
        elif 43 < age <= 53:
            file.at[i, 'age'] = '43-53'
        elif 53 < age <= 63:
            file.at[i, 'age'] = '53-63'
        elif 63 < age <= 73:
            file.at[i, 'age'] = '63-73'
        i += 1     
    return file        