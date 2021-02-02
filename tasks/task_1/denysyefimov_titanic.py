import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Let's calculate the percentage of
# male passengers who survived(men and women).

train_data = pd.read_csv("Your path")
train_data.head()
test_data = pd.read_csv("Your path")
test_data.head()

# End reading the datas and let's get
# started the machine learning. Train our model.

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100,
                               max_depth=5,
                               random_state=1
                               )
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Survived': predictions}
                      )
output.to_csv('my_submission.csv',
              index=False
              )
print("All is Done!")
