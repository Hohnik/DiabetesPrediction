<<<<<<< HEAD
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Transformer import ReplaceZeroWithMean


DATAPATH = "../Data/"
MODELPATHSKLEARN = "../Data/Models/SVM/SVM_sklearn/"


data = pd.read_csv(f"{DATAPATH}/diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

train_set, test_set, train_labels, test_labels = train_test_split(
    X, y, train_size=0.60, random_state=42
)

imputer = ReplaceZeroWithMean(["Glucose", "BloodPressure", "SkinThickness", "BMI"])
train_set = imputer.fit_transform(train_set)
test_set = imputer.transform(test_set)

scaler = StandardScaler()
train_set = scaler.fit_transform(train_set)
test_set = scaler.transform(test_set)


def save_model_as_sklearn(filename, parameters=None):
    parameters["max_iter"] = int(parameters["max_iter"])
    model = SVC(**parameters)
    model.fit(train_set, train_labels)
    joblib.dump(model, MODELPATHSKLEARN + f"{filename}.pkl")
||||||| c1bda8b
=======
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Transformer import ReplaceZeroWithMean


DATAPATH = '../Data/'
MODELPATHSKLEARN = '../Data/Models/SVM/SVM_sklearn/'



data = pd.read_csv(f"{DATAPATH}/diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

train_set, test_set, train_labels, test_labels = train_test_split(X, y, train_size=0.60, random_state=42)

imputer = ReplaceZeroWithMean(["Glucose", "BloodPressure", "SkinThickness", "BMI"])
train_set = imputer.fit_transform(train_set)
test_set = imputer.transform(test_set)

scaler = StandardScaler()
train_set = scaler.fit_transform(train_set)
test_set = scaler.transform(test_set)

def save_model_as_sklearn(filename, parameters=None):
    parameters['max_iter'] = int(parameters['max_iter'])
    model = SVC(**parameters)
    model.fit(train_set, train_labels)
    joblib.dump(model, MODELPATHSKLEARN + f'{filename}.pkl')
>>>>>>> d79c55d92c4bf99c2c8048bbcc5cb2918077f3a5
