# Bank Credit Score Prediction

# Dataset 
The [data](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) has been retrieved from Kaggle, and the below description table should be able to summarize the details about available variables, and the aim is to predict the credit score of creditors, and let’s see what we can do to achieve highest prediction accuracy.

![image](https://github.com/user-attachments/assets/f3d77050-77b4-44a6-88ce-adc1f2063201)

# EDA
## Missing Values
![image](https://github.com/user-attachments/assets/d6b826e0-2766-4677-a4ee-31f4681376bf)
## Training Data
![image](https://github.com/user-attachments/assets/4ab11267-ce0b-44d7-8852-78981a451e5b)
## Validation Data
![image](https://github.com/user-attachments/assets/b4ada1f0-e08e-427c-a810-01faffc8272e)

# Preprocessing
The preprocessing phase begins with understanding the variables, the corresponding data types and observing the quality of data, which is a complicated process given the fact that most of the variables potentially be in incorrect datatypes, with a lot of missing values and outliers.
![image](https://github.com/user-attachments/assets/6ac912d3-75db-44d5-aa35-e3438db5747b)

## Missing Value Treatment
The process of handling missing values and preprocessing in the credit score prediction project involved several key steps:

 * Missing Value Identification and Visualization: Using tools like missingno, the missing values were visualized in the dataset, identifying columns with missing data and incorrect values.

 * Handling Missing Values:

  * Name Column: Rows with missing values in the Name column were dropped, as no reasonable method could fill this data.
  * Numerical Columns: Missing or incorrect values in numerical columns like Annual_Income were replaced with the mean values of their respective columns.
  * Occupation Column: Missing values were replaced with "Unemployed."
  * Outlier Removal: Outliers in numerical columns were cleaned using the Interquartile Range (IQR) method, which removed extreme values that could skew the model results.
  * Data Type Corrections: Misclassified columns were converted to appropriate data types, such as converting strings with numeric information into actual numerical formats.

* Feature Engineering: For example, the Credit_History_Age column, which contained string values like "Years" and "Months," was transformed into a total number of months.

* These preprocessing steps ensured that the dataset was clean and ready for model building​

![image](https://github.com/user-attachments/assets/cd2ebb2d-1366-443f-b8d8-72d99a35339a)

# Ensemble Modelling - VotingClassifier
![image](https://github.com/user-attachments/assets/9691e2b3-65a8-4454-b3e3-f47427ff2d98)


```Python
 X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, shuffle=True)

# Instantiate individual classifiers
knn = KNeighborsClassifier(n_neighbors=3)
dt = DecisionTreeClassifier(max_depth=8, random_state=24)
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, max_features="log2", random_state = 24)

# Define a list of classifiers that contains (classifier_name, classifier)
classifiers=[("K Nearest Neighbors", knn),
             ("Decision Tree", dt),
             ("Random Forest", rf)]

for clf_name, clf in classifiers:
    # fit clf to the training set
    clf.fit(X_train, y_train)
    # predict the labels of the test set
    y_pred = clf.predict(X_test)
    # evaluate the accuracy of clf on the test set
    print("{:s} : {:.3f}".format(clf_name, accuracy_score(y_test, y_pred)))
```

## Hyperparameter Tuning
### DecisionTreeClassifier - MaxDepth 
```Python
training_acc={}
test_acc={}
max_depth_range=np.arange(1,20)
for max_d in max_depth_range:
    
    model = DecisionTreeClassifier(max_depth=max_d, random_state=42)
    model.fit(X_train, y_train)
    
    training_acc[max_d] = model.score(X_train, y_train)
    test_acc[max_d] = model.score(X_test, y_test)

# visualize test and training accuracies
fig,ax=plt.subplots()
ax.plot(max_depth_range, training_acc.values(), label="Training")
ax.plot(max_depth_range, test_acc.values(), label="Test")
plt.legend(["Training","Test"])
plt.title("DT Model Max_Depth Tuning")
plt.xlabel("Number of Max Depth")
plt.ylabel("Accuracy")
plt.axvline(x=8, linestyle="--")
plt.show()
```
![image](https://github.com/user-attachments/assets/e18440d3-551f-4451-9400-6439229c73eb)
![image](https://github.com/user-attachments/assets/8fa0b084-8337-449b-87c3-15e3563c1f16)

### KNN - N_neighbors
```Python
train_accuracies={}
test_accuracies={}
neighbors=np.arange(1,20)

for neighbor in neighbors:

    knn=KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

# Visualizing Model Complexity 
plt.figure(figsize=(8,6))
plt.title("'n_neighbors' Tuning")
plt.plot(neighbors, train_accuracies.values(), label="training accuracy"),
plt.plot(neighbors, test_accuracies.values(), label="test accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.axvline(x=3,linestyle="--")
plt.ylabel("Accuracy")
plt.show()
```
![image](https://github.com/user-attachments/assets/860313d4-6a5c-49f4-891d-1da95b965e34)


### RandomForestClassifier - N_Estimators
```Python
train_accuracies_rf={}
test_accuracies_rf={}

nestimate=np.arange(20,520,20)
for ns in nestimate:

    model_rf = RandomForestClassifier(n_estimators=ns, random_state = 24)
    model_rf.fit(X_train, y_train)
    
    train_accuracies_rf[ns] = model_rf.score(X_train,y_train)
    test_accuracies_rf[ns] = model_rf.score(X_test, y_test)

# Visualizing Model Complexity 
plt.figure(figsize=(8,6))
plt.title("'n_estimators' Tuning")
plt.plot(nestimate, train_accuracies_rf.values(), label="training accuracy"),
plt.plot(nestimate, test_accuracies_rf.values(), label="test accuracy")
plt.legend()
plt.xlabel("Number of Estimators")
plt.axvline(x=300,linestyle="--")
plt.ylabel("Accuracy")
plt.show()
```
![image](https://github.com/user-attachments/assets/52ebd872-49f0-4667-a77f-33483fdeab2a)

# Final Results

## Ensembling Best Hyperparameters
```Python
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, shuffle=True)

# Instantiate individual classifiers
knn = KNeighborsClassifier(n_neighbors=3)
dt = DecisionTreeClassifier(max_depth=8, random_state=24)
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, max_features="log2", random_state = 24)

# Define a list of classifiers that contains (classifier_name, classifier)
classifiers=[("K Nearest Neighbors", knn),
             ("Decision Tree", dt),
             ("Random Forest", rf)]

for clf_name, clf in classifiers:
    # fit clf to the training set
    clf.fit(X_train, y_train)
    # predict the labels of the test set
    y_pred = clf.predict(X_test)
    # evaluate the accuracy of clf on the test set
    print("{:s} : {:.3f}".format(clf_name, accuracy_score(y_test, y_pred)))

# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers)

# fit 'vc' to the training set and predict test set labels
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)

# evaluate the test-set accuracy of 'vc'
print("Voting Classifier: {}".format(round(accuracy_score(y_test, y_pred),3)))

```
### Accuracy
Voting Classifier : 0.771, was 0.762 before the tuning.
