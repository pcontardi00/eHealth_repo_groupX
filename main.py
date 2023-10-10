# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev

# Importing the csv
df = pd.read_csv("dataset_project_eHealth20232024.csv")
df.info()
print(df.shape)

print(df.head(5))

# Exploratory data analysis
# divide the three categories of questions
df_soceco5 = df.iloc[:,0:5]
print(df_soceco5.head())

df_phq9 = df.iloc[:, 5:14]
print(df_phq9.head())

df_gad7 = df.iloc[:, 14:21]
print(df_gad7.head())

df_ehea8 = df.iloc[:, 21:29]
print(df_ehea8.head())

df_hogg13 = df.iloc[:, 29:42]
print(df_hogg13.head())

df_ccsq12 = df.iloc[:, 42:]
print(df_ccsq12.head())



# Data preparation: drop data, substitute with eventual others
df_new = pd.concat([df_soceco5, df_ehea8], axis=1)
df_2 = df_new.dropna()

# da assegnare titoli attributi

# Trasformazioni
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

df_categorical = df_2.iloc[:, 1:3]
encoder = OneHotEncoder(sparse = False)
scaler = StandardScaler()

df_encoded = pd.DataFrame(encoder.fit_transform(df_categorical))

print(df_encoded.shape)


# age, marital status, gender, education, income
df_numerical = pd.concat([df_2.iloc[:, 0],df_2.iloc[:, 3:]], axis=1)
df_standardized = pd.DataFrame(scaler.fit_transform(df_numerical))

# outlier identification
age = df_numerical.iloc[:,0].tolist()
income = df_numerical.iloc[:, 3].tolist()

def get_statistics(my_list: list[float]) -> tuple:
    # We use statistics to obtain mean and std, while numpy will be used for median and percentiles.
    # Notice that we do not call our variable mean, as that would overwrite the function that we are using.
    l_mean = mean(my_list)
    l_std = stdev(my_list)
    l_median = np.percentile(my_list, 50)
    l_first_percentile = np.percentile(my_list, 25)
    l_third_percentile = np.percentile(my_list, 75)

    print(f"Mean: {l_mean}\n"
          f"Standard Deviation: {round(l_std, 3)}\n"
          f"Median: {l_median}\n"
          f"Percentiles: {l_first_percentile}, {l_third_percentile}")

    return l_first_percentile, l_third_percentile


# Call the function
percentiles_a = get_statistics(age)
percentiles_b = get_statistics(income)

# 2. Use a statistical method to identify outliers
# We have to perform this action twice, so let's get a function here too.
def get_outliers(my_list: list[float], percentiles: tuple):
    # Define the interquartile range.
    iqr = percentiles[1] - percentiles[0]

    # Create an empty list to store outliers.
    outliers = []
    # This for cycle runs along all elements of our list my_lost, taking one at a time and calling it "element".
    for element in my_list:
        # Check if the current element in our list is below or above a certain threshold.
        # If it is, add it to the outliers list.
        if (element < percentiles[0] - 1.5 * iqr) or (element > percentiles[1] + 1.5 * iqr):
            outliers.append(element)

    print(f"Outliers: {outliers}")

get_outliers(age, percentiles_a)
get_outliers(income, percentiles_b)

# Concatenating the two dataframes
df_end = pd.concat([df_encoded, df_standardized], axis = 1)
print(df_end.head())

# Outliers detection
plt.boxplot(df_end.iloc[:,12])
plt.show()

# PCA
import prince
pca = prince.PCA(
    n_components = 5,
    random_state=42
)
pca = pca.fit(df_end)

pca.eigenvalues_summary


# Clustering
from sklearn_extra.cluster import KMedoids
k = 3
kmedoids = KMedoids(k,random_state= 0).fit(df_end)
print(kmedoids.labels_)