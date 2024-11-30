# Machine-Learning I

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
