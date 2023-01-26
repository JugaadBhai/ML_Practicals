from operator import imod
import random as r
from numpy import append
from sklearn.linear_model import LinearRegression

print("Ready the training data")
print("=============================================")

feature_set = []
training_set = []
no_of_rows = 200
limit = 2000

for i in range(0, no_of_rows):
  x = r.randint(0, limit)
  y = r.randint(0, limit)
  z = r.randint(0, limit)
  g = (10 * x) + (2 * y) + (3 * z)

  print("=======================================================")
  print("X = ", x)
  print("Y = ", y)
  print("Z = ", z)
  print("G = ", g)
  print("=======================================================\n")

  feature_set.append([x,y,z])
  training_set.append(g)

print("====== Training of Model Started =======")
model = LinearRegression()
model.fit(feature_set, training_set)
print("====== Training of Model Ended =======\n")

print("====== Testing of Model Started =======")
print("====== Enter the testing data =======")
x = int(input("X = "))
y = int(input("Y = "))
z = int(input("Z = "))
test_data = [[x,y,z]]
prediction = model.predict(test_data)

print("Prediction: " + str(prediction) + "\t" + "Coefficient: " + str(model.coef_))

