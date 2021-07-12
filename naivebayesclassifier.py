from csv import reader
from scipy import stats
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

def read_csv(filename):
    '''Reads the CSV from the given filename'''
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def extract_variables(dataset, idx, datatype):
    '''Extracts variables from columns in dataset and creates new lists of variables'''
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

# Point biserial correlation to test for correlation between binary and continuous variables
print('age, test result: {0}'.format(stats.pointbiserialr(test_result, age)))
print('temperature, test result: {0}'.format(stats.pointbiserialr(test_result, temperature)))
print('bmi, test result: {0}'.format(stats.pointbiserialr(test_result, bmi)))

# Matthews correlation for binaries
print('onset, test result: {0}'.format(metrics.matthews_corrcoef(onset, test_result)))
print('myalgia, test result: {0}'.format(metrics.matthews_corrcoef(myalgia, test_result)))
print('chills, test result: {0}'.format(metrics.matthews_corrcoef(chills, test_result)))
print('cough, test result: {0}'.format(metrics.matthews_corrcoef(cough, test_result)))
print('fever, test result: {0}'.format(metrics.matthews_corrcoef(fever, test_result)))

# combine features in as list of lists
# concat = zip(age, onset, myalgia, temperature, chills, cough, bmi, fever)
concat = zip(onset, myalgia, chills, cough, fever)
features = list(concat)
# print(features)

# test if data set is balanced  
print(stats.itemfreq(test_result))

# create training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, test_result, test_size=0.3, random_state=42)


# create Gaussian Naive Bayes model
model = GaussianNB()

# train the model
model.fit(X_train, y_train)

# the number of parameters in the model should be the same as in logistic regression sind the same vareables are used

# predict the response for test dataset
y_pred = model.predict(X_test)

# evaluate accuracy
print('Accuracy Gaussian Naive Bayes: {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('mse Gaussian Naive Bayes: {}'.format(metrics.mean_squared_error(y_test, y_pred)))



# create Decision Tree model
model = tree.DecisionTreeClassifier()

# train the model
model.fit(X_train, y_train)

# predict the response for test dataset
y_pred = model.predict(X_test)

# evaluate accuracy
print('Accuracy Decision Tree: {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('mse Decision Tree: {}'.format(metrics.mean_squared_error(y_test, y_pred)))



# create Logistic Regression model
model = LogisticRegression()

# train the model
model.fit(X_train, y_train)

# num of parameters in model
num_params = len(model.coef_) + 1
print('Number parameter Logistic Regresssion: {0}'.format(num_params))

# predict the response for test dataset
y_pred = model.predict(X_test)

# evaluate accuracy
print('Accuracy Logistic Regression: {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('mse Logistic Regression: {}'.format(metrics.mean_squared_error(y_test, y_pred)))



# create k-Nearest Neighbors model
model = KNeighborsClassifier(n_neighbors=3)

# train the model
model.fit(X_train, y_train)

# predict the response for test dataset
y_pred = model.predict(X_test)

# evaluate accuracy
print('Accuracy k-Nearest Neighbors: {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('mse k-Nearest Neighbors: {}'.format(metrics.mean_squared_error(y_test, y_pred)))



# create SVM model
model = svm.SVC()

# train the model
model.fit(X_train, y_train)

# predict the response for test dataset
y_pred = model.predict(X_test)

# evaluate accuracy
print('Accuracy SVM: {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('mse SVM: {}'.format(metrics.mean_squared_error(y_test, y_pred)))