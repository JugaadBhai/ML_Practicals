import csv

num_attributes = 6

dataset = []

print("============= TRAINING DATASET ================")

with open("D:\\MSc IT\\PART 2\\SEM 3\\ML Practicals\\Practical 2\\training.csv","r") as csvfile:
  readr = csv.reader(csvfile)
  for row in readr:
    dataset.append(row)
    print(row)

print("\n The initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes
print(hypothesis)

for j in range(0,num_attributes):
  hypothesis[j] = dataset[1][j]
  print("\n Find S: Finding a Maximally Specific Hypothesis\n")
  
  for i in range(1,len(dataset)):
    if dataset[i][num_attributes] == 'Yes':
      for j in range(0, num_attributes):
        if dataset[i][j] != hypothesis[j]:
          hypothesis[j] = '?'
        else:
          hypothesis[j] = dataset[i][j]
  
  print(" For Training instance No:{0} the hypothesis is ".format(i),hypothesis)
print("\n The Maximally Specific Hypothesis for a given Training Examples :\n")
print(hypothesis)
