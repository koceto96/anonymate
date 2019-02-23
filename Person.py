class Person:
    def __init__(self, name, postcode, tel, age, income):
        self.name = name
        self.postcode = postcode
        self.tel = tel
        self.age = age
        self.income = income
        
    def __str__(self):
        return(f"{self.name}, {self.postcode}, {self.tel}, {self.age}, {self.income}")
        