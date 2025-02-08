# Recommendation-System


###**Problem Statement**


- **Description:** Build a recommendation system for products or content
based on user behaviour and preferences. This enhances user
experience by suggesting relevant items.
- **Why:** Recommendation systems increase user engagement and sales
by providing personalized recommendations.
- **Tasks:**
    ▪ Collect user interaction data.

    ▪ Example datasets Click Here

    ▪ Preprocess data (normalization, handling missing values).

    ▪ Apply recommendation algorithms (collaborative filtering, contentbased filtering).

    ▪ Evaluate recommendations and improve the model.

Loading files to google colab

from google.colab import files
uploaded = files.upload()

importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fashion = pd.read_csv("fashion_products.csv")

fashion.head()

fashion.info()

fashion.isnull().sum().sum()

fashion.duplicated().sum()

#finding the outliers
for i in fashion.columns:
  if fashion[i].dtype != 'object':
    sns.boxplot(fashion[i])
    plt.show()

fashion.columns

Encoding the columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Cols_to_encode = ['Product Name', 'Brand', 'Color', 'Category']

for col in Cols_to_encode:
  fashion[col] = le.fit_transform(fashion[col])

fashion.drop('Size', axis = 1, inplace = True)

fashion

Splitting the data set

from sklearn.model_selection import train_test_split

X = fashion.drop('Rating', axis=1)  # Replace 'target_column' with your actual target column name
y = fashion['Rating']

# Assuming your data is in a pandas DataFrame called 'fashion'
train_data, test_data = train_test_split(fashion, test_size=0.2, random_state=42)  # Adjust test_size and random_state as needed

Applying StandardScaling

from sklearn.preprocessing import StandardScaler

# Assuming your features are in a DataFrame called 'X'
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

installing Surprise library

! pip install scikit-surprise

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Assuming 'user_id', 'product_id', and 'rating' are columns in your dataset
reader = Reader(rating_scale=(1, 5))  # Adjust rating scale if needed
fashion = Dataset.load_from_df(fashion[['User ID', 'Product ID', 'Rating']], reader)

trainset, testset = train_test_split(fashion, test_size=.25)
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)

from surprise import accuracy

# For collaborative filtering:
rmse = accuracy.rmse(predictions)





