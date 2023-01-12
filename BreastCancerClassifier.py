from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#Load data into variable
breast_cancer_data = load_breast_cancer()
#Split data for training and validation
data_train, data_test, label_train, label_test = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)
#creating list of K values and empty list where we will add validation scores
k_list = [ * range(1,101)]
accuracies = []
#generating validation score for K from 1 to 100 and saving scores in accuracies list
for k in range (1,101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(data_train,label_train)
  accuracies.append (classifier.score(data_test,label_test))
#Print best score K values
max_k_list = [index for index, item in enumerate(accuracies) if item == max(accuracies)]
print(max_k_list)
#Generating Graph of K and Score(accuracy of the model on on specific K value)
plt.plot(k_list,accuracies)
plt.xlabel("Neighbor value 'K'")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()