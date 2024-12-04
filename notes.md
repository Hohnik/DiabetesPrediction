# Machine-Learning I


Notes:
Try to visualize the tensor of score values of the grid search results of all models (if possible to get all the value of the grid search scores)

## Dataset
Our dataset is about "Diabetes prediction". 
It has 8 features and 1 label. 
Its a binary classification problem.


## Support Vector Machines 
Average time per trained model = 5.3s

### SupportVectorClassification with Hyperparameter Tuning and Cross validation (time: 5.38min, permutations: 50)
```python
params = {
    "kernel": ["linear"], 
    "C": [0.1, 1, 10, 100, 1000]
    }
cv_folds = 5
estimator = SVC(C=0.1, kernel='linear', random_state=37)
train_score = 0.7707317073170732
test_score = 0.7532467532467533 
```

### SupportVectorClassification with Hyperparameter Tuning and Cross validation (time: 30min, permutations: 350)
```python
params = {
    "kernel": ["linear", "rbf"], 
    "C": [0.1, 1, 10, 100, 1000], 
    "gamma": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }
cv_folds = 5
estimator = SVC(C=1, gamma=0.001, kernel='linear', random_state=37)
train_score = 0.7772357723577237
test_score = 0.7987012987012987 
```

### SupportVectorClassification with cuml instead of sklearn to speed up the training
I think we have different approach to the hyperparameter tuning, so the results are not comparable. 
but the training time is much faster. Time is 3.04 seconds for 50 permutations (LinearSVC).
->  Best parameters: {'C': np.float64(205.0111588307334), 'penalty': 'l2'}
    Accuracy: 0.75
    Time taken: 3.04 seconds

-> Best parameters: {'C': np.float64(831.1616457652171), 'penalty': 'l2'}
    Accuracy: 0.74
    Time taken: 15.45 seconds

last results in order:

Best parameters: {'C': 100, 'gamma': 0.001}
Accuracy: 0.77
Time taken: 9.41 seconds

Best parameters: {'C': 0.1, 'penalty': 'l2'}
Accuracy: 0.75
Time taken: 1.77 seconds

Best parameters: {'C': np.float64(822.009328781485)}
Accuracy: 0.74
Time taken: 17.45 seconds


## Random Forest
Made first Model quick and dirty, no optimization or anything.