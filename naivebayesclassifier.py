from csv import reader
from scipy import stats

# Read csv file
def read_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def extract_variables(dataset, idx, datatype):
    new_list = []
    for s in dataset:
        if datatype == 'integer':
            variable = int(s[idx])
        else:
            variable = float(s[idx])
        new_list.append(variable)
    return new_list







filename = 'Flu_data.csv'
dataset = read_csv(filename)

# description of categories of different columns in dataset:
# 0: 'Id' (unique identifier for each individual)
# 1: 'Age yrs' (int)
# 2: 'Onset' (< 48hrs: 0, >= 48hrs: 1)
# 3: 'Myalgia' (binary)
# 4: 'Temperature C' (float)
# 5: 'Chills' (binary)
# 6: 'Cough' (binary)
# 7: 'BMI' (float)
# 8: 'Fever' (binary)
# 9: 'Test Result' (binary)

# create list of columns of dataset
age = extract_variables(dataset, 1, 'integer')
onset = extract_variables(dataset, 2, 'integer')
myalgia = extract_variables(dataset, 3, 'integer')
temperature = extract_variables(dataset, 4, 'float')
chills = extract_variables(dataset, 5, 'integer')
cough = extract_variables(dataset, 6, 'integer')
bmi = extract_variables(dataset, 7, 'float')
fever = extract_variables(dataset, 8, 'integer')
test_result = extract_variables(dataset, 9, 'integer')

print(stats.pointbiserialr(test_result, age))
print(stats.pearsonr(chills, test_result))


