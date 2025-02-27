
#@title Run this to download your data! { display-mode: "form" }
# Load the data!
import pandas as pd
from sklearn import metrics

data = pd.read_csv('logistic_Regression\cancer.csv')
data['diagnosis'].replace({'M':1, 'B':0}, inplace = True)
data.to_csv('cancer.csv')
del data

"""## Loading our annotated dataset

The first step in building our breast cancer tumor classification model is to load in the dataset we'll use to "teach" (or "train") our model.
"""

# First, import helpful Python tools for loading/navigating data
import os             # Good for navigating your computer's files
import numpy as np    # Great for lists (arrays) of numbers
import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv)
from sklearn.metrics import accuracy_score   # Great for creating quick ML models

data_path  = 'cancer.csv'

# Use the 'pd.read_csv('file')' function to read in read our data and store it in a variable called 'dataframe'
dataframe = pd.read_csv(data_path)

dataframe = dataframe[['diagnosis', 'perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean']]
dataframe['diagnosis_cat'] = dataframe['diagnosis'].astype('category').map({1: '1 (malignant)', 0: '0 (benign)'})

"""# Exploring our data

## Looking at our dataset
  
 A key step in machine learning (and coding in general!) is to view the structure and dimensions of our new dataframe, which stores all our training data from the tumor biopsies. You can think of dataframes like Google or Microsoft Excel spreadsheets (large tables with row/column headers).

We want to confirm that the size of our table is correct, check out the features present, and get a more visual sense of what it looks like overall.

**Use the '.head()' method to show the first five rows of the table and their corresponding column headers (our biopsy features!)**
"""

# YOUR CODE HERE:
dataframe.head()
# END CODE

"""Our colleague has given us documentation on what each feature column means. Specifically:

* $diagnosis$: Whether the tumor was diagnosed as malignant (1) or benign (0).
* $perimeter$_$mean$: The average perimeter of cells in that particular biopsy
* $radius$_$mean$: The average radius of cells in that particular biopsy
* $texture$_$mean$: The average texture of cells in that particular biopsy
* $area$_$mean$: The average area of cells in that particular biopsy
* $smoothness$_$mean$: The average smoothness of cells in that particular biopsy
* $concavity$_$mean$: The average concavity of cells in that particular biopsy
* $symmetry$_$mean$: The average symmetry of cells in that particular biopsy

Recall that the term mean refers to taking an average (summing the values for each cell and dividing by the total number of cells observed in that biopsy).
"""

# Next, we'll use the 'info' method to see the data types of each column
dataframe.info()

"""**Discussion Question:** Which columns are numeric? Why?

## Visualizing our dataset

How can we determine the relationship between each of the "features" of these cells and the diagnosis?

The best way is to graph certain features in our data and see how they vary between different diagnoses! We will use some Python libraries like Seaborn and Matplotlib to make this an easier task for us.
"""

# First, we'll import some handy data visualization tools
import seaborn as sns
import matplotlib.pyplot as plt

"""Let's focus on one feature for now: mean radius. How well does it predict diagnosis?"""

sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', data = dataframe, order=['1 (malignant)', '0 (benign)'])
dataframe.head()

"""**Discussion Question:** How would you interpret what is going on in the chart above?

**Exercise**: Try out some other features (e.g. perimeter_mean, texture_mean, smoothness_mean) to see how they relate to the diagnosis. Which single feature seems like the best predictor?

# Predicting Diagnosis

Let's start by predicting a diagnosis using a single feature: radius mean.

## Approach 1: Can we use linear regression to classify these cells?

Let's start by using an algorithm that we've seen before: linear regression!

**Discussion Question: How might linear regression be useful to classify examples from this dataset?**
"""

#@title Run this to fit and visualize a linear regression (double-click to see code!)
from sklearn import linear_model

X,y = dataframe[['radius_mean']], dataframe[['diagnosis']]

model = linear_model.LinearRegression()
model.fit(X, y)
preds = model.predict(X)

sns.scatterplot(x='radius_mean', y='diagnosis', data=dataframe)
plt.plot(X, preds, color='r')
plt.legend([ 'Data', 'Linear Regression Fit'])

#@title Take a look at the linear regression model and answer the following questions:

#@markdown What does a diagnosis of 0.0 mean?
diagnosis_0 = "Benign" #@param ["Malignant", "Benign", "Choose An Answer"]

#@markdown What does a diagnosis of 1.0 mean?
diagnosis_1 = "Malignant" #@param ["Malignant", "Benign", "Choose An Answer"]

if diagnosis_0 == 'Benign' and diagnosis_1 == 'Malignant':
  print("Correct! 0.0 is a benign prediction and 1.0 is malignant.")
else:
  print("One or both of our diagnoses' interpretations is incorrect. Try again!")

"""**Discuss: Did this linear regression model do well?**

Hint: What would our linear regression model predict for a mean radius of 25? How about 30? Is this an appropriate output?

##Approach 2: Classification -  Simple Boundary Classifier
The variable we are trying to predict is categorical, not continuous! So we can't use a linear regression; we have to use a classifier.

### Classification is just drawing boundaries!

The simplest approach to classification is just drawing a boundary. Let's pick a boundary value for the radius mean and see how well it separates the data.
"""

boundary = 17.5 # change me!

sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', data = dataframe, order=['1 (malignant)', '0 (benign)'])
plt.plot([boundary, boundary], [-.2, 1.2], 'g', linewidth = 2)

"""**Question:** Does this boundary value separate the data well? What do the points in each part of the graph represent?

### Building the boundary classifier

Here we build a boundary classifier function that takes in a **target boundary**: a particular value of radius mean. This function will take in a boundary value of our choosing and then classify the data points based on whether or not they are above or below the boundary.

**Exercise: Write a function to implement a boundary classifier.** You'll take in a `target_boundary` (a `float` or `int` like 15) and a `radius_mean_series` (a list of values) and return a list of predictions!
"""

def boundary_classifier(target_boundary, radius_mean_series):
  result = [] #fill this in with predictions!
  # YOUR CODE HERE
  for radius in radius_mean_series:
    if radius<target_boundary:
      result.append(0)
    else:
      result.append(1)
  return result

"""The code below chooses a boundary and runs your classifer."""

chosen_boundary = 15.3 #Try changing this!

y_pred = boundary_classifier(chosen_boundary, dataframe['radius_mean'])
dataframe['predicted'] = y_pred

y_true = dataframe['diagnosis']

sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', hue = 'predicted', data = dataframe, order=['1 (malignant)', '0 (benign)'])
plt.plot([chosen_boundary, chosen_boundary], [-.2, 1.2], 'g', linewidth = 2)

"""What do you think of the results based on the graph?

We can take a look at `y_true` and `y_pred` - how similar do they look?
"""

print (list(y_true))
print (y_pred)

"""Let's calculate our accuracy!"""

accuracy = accuracy_score(y_true,y_pred)
print(accuracy)

"""**Now adjust the chosen boundary above to get the best possible 'separation'.** As you do that, think about what it means for a separation to be 'good' - is it just the highest accuracy?

##Approach 3: Logistic Regression - using machine learning to determine the optimal boundary

Now, it's time to move away from our simple guess-and-check model and work towards implementing an approach that can automatically find a better separation. One of the most common methods for this is called 'Logistic Regression'.

### Training Data vs Test Data

We'll split up our data set into groups called 'train' and 'test'. We teach our 'model' the patterns using the train data, but the whole point of machine learning is that our prediction should work on 'unseen' data or 'test' data.

The function below does this for you.
"""

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1)

"""Let's now take a look at the 'train' and 'test' groups:

"""

print('Number of rows in training dataframe:', train_df.shape[0])
train_df.head()

print('Number of rows in test dataframe:', test_df.shape[0])
test_df.head()

"""### Single Variable Logistic Regression
To start with, let's set our input feature to be radius mean and our output variable to be the diagnosis.

We will use this to build a logistic regression model to predict the diagnosis using radius mean.
"""

X = ['radius_mean']
y = 'diagnosis'

X_train = train_df[X]
print('X_train, our input variables:')
print(X_train.head())
print()

y_train = train_df[y]
print('y_train, our output variable:')
print(y_train.head())

"""**Discuss:** What's the difference between X_train and y_train?

Now, let's prepare our model (we haven't trained it yet):
"""

# Here, we create a 'reg' object that handles the line fitting for us!
logreg_model = linear_model.LogisticRegression()

"""###Making Predictions

Next, we want to tell our 'logreg_model' object to take in our inputs (X) and our true labels (y) and fit a line that predicts y from X.

**Exercise:** Can you place the arguments `X_train` and `y_train` correctly into this function to do this?

`logreg_model.fit(FILL_ME_IN, FILL_ME_IN)`

"""

### YOUR CODE HERE
logreg_model.fit(X_train,y_train)
### END CODE

"""### Testing our model

How do we know if our 'model' is actually 'learning' anything? We need to test it on unseen data.

Here we will be designating test inputs to check our model. Let's prepare the inputs and outputs from our testing dataset - try printing them out!
"""

X_test = test_df[X]
y_test = test_df[y]

"""### Making predictions on our test set

Next, we need to figure out what our line thinks the diagnosis is based on our data points

**Exercise:** Fill in the appropriate input to this function and run the function below.

`y_pred = logreg_model.predict(FILL_ME_IN)`
"""

## YOUR CODE HERE
y_pred=logreg_model.predict(X_test)
## END CODE

"""Run the code below to visualize the results!"""

test_df['predicted'] = y_pred.squeeze()
sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', hue = 'predicted', data=test_df, order=['1 (malignant)', '0 (benign)'])

"""How does it look compared to the predictions before?

### Finally, let's evaluate the accuracy of our model.
"""

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

"""## What is logistic regression doing? It's giving 'soft' predictions!

"""

#@title Run this to plot logistic regression's soft probabilities { display-mode: "form" }

# Let's visualize the probabilities for `X_test`
y_prob = logreg_model.predict_proba(X_test)
X_test_view = X_test[X].values.squeeze()
plt.xlabel('radius_mean')
plt.ylabel('Predicted Probability')
sns.scatterplot(x = X_test_view, y = y_prob[:,1], hue = y_test, palette=['purple','green'])

"""The Y-axis is the  probability of being 'malignant' and the X-axis is the radius mean. The colors show the **true** diagnosis (this is different than previous graphs!)

**Can you interpret or take a guess about what the graph above is saying?**

# Approach 4: Multiple Feature Logistic Regression

Which features best predict the diagnosis?

Now that we can use logistic regression to find the optimal classification boundary, let's try out other features to see how well they predict the diagnosis.

First let's print out one row of our table so we can see what other features we have available to us.
"""

dataframe.head(1)

"""### Experimenting with Single-Variable Logistic Regression

First, let's practice what we've done already! Fill in the code below to prepare your X and y data, fit the model on the training data, and predict on the test data.

**Exercise:** Once you have this code working, try replacing radius_mean with other features to see how well each feature predicts diagnosis!
"""

X = ['radius_mean'] #Try changing this later!
y = 'diagnosis'

# 1. Split data into train and test
train_df, test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1)

# 2. Prepare your X_train, X_test, y_train, and y_test variables by extracting the appropriate columns:

# 3. Initialize the model object

# 4. Fit the model to the training data

# 5. Use this trained model to predict on the test data

# 6. Evaluate the accuracy by comparing to to the test labels and print out accuracy.

"""**Discussion**: Which features best predicted diagnosis? What does this teach us about breast cancer?

## Can we use multiple features together to do even better?
So far, we've just been using `radius_mean` to make predictions. But there's lots of other potentially important features that we could be using!

Let's take a look again:
"""

dataframe.head(1)

"""### Logistic Regression with Multiple Features

Now, let's try re-fitting the model using **your choice of multiple features.**

Just add more features to the list: for example, to use two features you could have

`X = ['radius_mean','area_mean']`
"""

X = ['radius_mean']
y = 'diagnosis'

# 1. Split data into train and test
train_df, test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1)

# 2. Prepare your X_train, X_test, y_train, and y_test variables by extracting the appropriate columns:

# 3. Initialize the model object

# 4. Fit the model to the training data

# 5. Use this trained model to predict on the test data

# 6. Evaluate the accuracy by comparing to to the test labels and print out accuracy.

"""Logistic Regression can learn an optimal classification boundary by using multiple features together, which can improve its prediction accuracy even more!

# Bonus Discussion: What makes a separation good?

We know our overall accuracy, so we know how many errors we make overall.

But errors come in two kinds:

**False positives:** The model predicts that a sample is malignant (positive), but it's actually benign.

**False negatives:** The model predicts that a sample is benign (negative), but it's actually malignant.

**Discuss:** In medical diagnoses, what are the dangers of each kind of mistake? What kind is worse? Can you think of an application where the opposite is true?

A key insight is that there's a trade-off between the two kinds of errors! For example, how could you make a classifier that's guaranteed to have no false negatives? Would that be a good classifier?

We have to find an acceptable balance!

###Confusion Matrices
Next, let's evaluate the performance of our model quantitatively. We can visualize statistics on the number of correct vs. incorrect predictions using a confusion matrix that shows the following:

![Confusion Matrix](https://miro.medium.com/max/860/1*7EcPtd8DXu1ObPnZSukIdQ.png)

where the terms mean:

* **TP (True Positive)** = The model predicted positive (malignant in our case, since malignant has a label of 1) and it’s true.
* **TN (True Negative)** = The model predicted negative (benign in our case, since benign has a label of 0) and it’s true.
* **FP (False Positive)** = The model predicted positive and it’s false.
* **FN (False Negative)** = The model predicted negative and it’s false.
"""

#@title Run this code to create a confusion matrix. { display-mode: "form" }
#@markdown If you are curious how it works you may double-click to inspect the code.

# Import the metrics class
from sklearn import metrics

# Create the Confusion Matrix
# y_test = dataframe['diagnosis']
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Visualizing the Confusion Matrix
class_names = [0,1] # Our diagnosis categories

fig, ax = plt.subplots()
# Setting up and visualizing the plot (do not worry about the code below!)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g') # Creating heatmap
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y = 1.1)
plt.ylabel('Actual diagnosis')
plt.xlabel('Predicted diagnosis')

#@title Take a look at the confusion matrix and answer the following questions:

#@markdown What are the values in the top left (0, 0)?
top_left = "Choose an Answer" #@param ["True Positives", "True Negatives", "False Positives", "False Negatives", "Choose an Answer"]

#@markdown What are the values in the bottom right (1, 1)?
bottom_right = "Choose an Answer" #@param ["True Positives", "True Negatives", "False Positives", "False Negatives", "Choose an Answer"]

#@markdown What are the values in the top right (0, 1)?
top_right = "Choose an Answer" #@param ["True Positives", "True Negatives", "False Positives", "False Negatives", "Choose an Answer"]

#@markdown What are the values in the bottom left (1, 0)?
bottom_left = "Choose an Answer" #@param ["True Positives", "True Negatives", "False Positives", "False Negatives", "Choose an Answer"]

if top_left == "True Negatives" and bottom_right == "True Positives":
  print("Correct! Our results are True if our model is correct!")
else:
  print("One or both of our (0, 0) and (1, 1) interpretations is incorrect. Try again!")

if top_right == "False Positives":
  print("Correct! A false positive is when our model predicts that a sample is malignant when it's actually benign.")
else:
  print("That's not quite what (0, 1) values are. Try again!")

if bottom_left == "False Negatives":
  print("Correct! A false negative is when our model predicts that a sample is benign when it's actually malignant.")
else:
  print("That's not quite what (1, 0) values are. Try again!")

"""**Discuss:**
- How many `True` values did our model predict?
- How many `False` values?
- Is our model a good classifier? Why or why not?

###Optional Challenge Exercise: Choosing a Metric

Depending on the situation, we might measure success in different ways. For example, we might use:

**Accuracy:** What portion of our predictions are right?

**Precision:** What portion of our positive predictions are actually positive?

**Recall:** What portion of the actual positives did we identify?

**Discuss: Which metric is most important for cancer diagnosis?**

To calculate any of these, we can use the numbers from our confusion matrix:
"""

print (cnf_matrix)
(tn, fp), (fn, tp) = cnf_matrix
print ("TN, FP, FN, TP:", tn, fp, fn, tp)

"""Now, calculate your model's performance by your chosen metric! You can use the [table on Wikipedia ](https://en.wikipedia.org/wiki/Confusion_matrix) to choose a metric and find a formula. How does it change your view of your model?

"""

#YOUR CODE HERE

"""**Congratulations!** You've successfully trained and evaluated a logistic regression model for diagnosing cancer.

#Optional: Decision Trees Walkthrough

Finally, let's try a different classification model: decision trees! Recall that with decision trees, we choose features that create the best splits of our dataset (separates it into classes as best it can at that time).
"""

#@title Create the model { display-mode: "both" }
from sklearn import tree

# We'll first specify what model we want, in this case a decision tree
class_dt = tree.DecisionTreeClassifier(max_depth=3)

# We use our previous `X_train` and `y_train` sets to build the model
class_dt.fit(multi_X_train, y_train)

#@title Visualize and interpret the tree
plt.figure(figsize=(13,8))  # set plot size
tree.plot_tree(class_dt, fontsize=10)

#@title Find the predictions based on the model { display-mode: "both" }
# now let's see how it performed!
multi_y_pred = class_dt.predict(multi_X_test)

#@title Calculate model performance { display-mode: "both" }
print("Accuracy: ", metrics.accuracy_score(y_test, multi_y_pred))
print("Precision: ", metrics.precision_score(y_test, multi_y_pred))
print("Recall: ", metrics.recall_score(y_test, multi_y_pred))

"""**Question: What features are included in this classifier? How might you interpret this tree? Did this do better than the logistic regression?**

# Advanced (Optional): Choosing a Classifier
We've studied two common classifiers, but many more are available. You can read about some of them [here](https://stackabuse.com/overview-of-classification-methods-in-python-with-scikit-learn/).

Let's try to choose the overall best classifier for this dataset. Fill in the code below to:
*   Use a for loop to train and evaluate each classifer in the list on our dataset.
*   Calculate the precision, recall, and accuracy on the test set for each classifier, and store the results in a data frame so it's easy to analyze.
*   Create plots to show the relationships between precision, accuracy, and recall and help you choose the "best" classifier.

Then experiment with changing the hyperparameters (options) of each classifier - can you get even better results?
"""

#@title Run this to import classifiers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#Once you've got your code working, try changing the hyperparameters of the classifiers
#to see if you can get even better results.
#Can you find out what the hyperparameters mean?
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


#Use a for loop to train and test each classifier, and print the results
#You might find the code above useful, as well as https://towardsdatascience.com/a-python-beginners-look-at-loc-part-2-bddef7dfa7f2 .

### YOUR CODE HERE ###




### END CODE ###



#Using pyplot, show the relationships between precision, recall, and/or accuracy.
#Tutorial here: https://matplotlib.org/tutorials/introductory/pyplot.html

### YOUR CODE HERE ###




### END CODE ###

"""**Think about:**
*   Which classifier would you choose?
*   What are the relationships among precision, recall, and accuracy? For this dataset, which is most important?
*   Can you find more successful hyperparameters for each classifer?

Your experiments will help you find a classifier that works very well on our test set. However, you're running a risk by doing so much manual fine-tuning: you might end up "overfitting" (on a more meta level) by choosing a classifier that works well on your test set, but might not work well on other data.

That's why most machine learning projects actually use [*three* datasets](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7): a training set that we use to train each candidate model; a validation set that we use to evaluate each candidate model and choose the best one; and finally, a test set which we use only once, to report the overall performance of our project.



"""