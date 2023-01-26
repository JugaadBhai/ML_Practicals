# Library Required For PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Reading Dataset from iris.data which is available in pandas library

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length','sepal-width','petal-length','petal-width','Class']
dataset = pd.read_csv(url, names = names)

# Display iris.data Dataset

print(dataset.head())
x = dataset.drop('Class', 1)
y = dataset['Class']

# Splitting the dataset into the Training set and Test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Display Training and Testing data

print("Dataset before PCA")
print(x_train)
print(x_test)

# Creating PCA object

pca = PCA()
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# Giving a Principal feature to model

pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print("Dataset after PCA")
print(x_train)
print(x_test)