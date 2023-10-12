
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev

# Importing the csv
df = pd.read_csv(r"C:\Users\irebe\OneDrive - Politecnico di Milano\Documenti\polimi\5 anno 1 semestre\E-heath\project\Project Data\dataset_project_eHealth20232024.csv")
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


#I build a dataframe with socioeconomic, eco-Anxiety e depression data
df_new = pd.concat([df_soceco5, df_hogg13, df_phq9], axis=1)
df.info()
print(df_new.shape)

# provo a droppare i valori nulli
df_2 = df_new.dropna()
print(df_2.shape)  # da 160 righe ne rimangono 140 ci può stare?

#qui ho provato a vedere se c'erano righe duplicate
# df_3 = df_new.drop_duplicates()
# print(df_3.shape)
# df_4 = df_new.dropna()
# print(df_4.shape)

# TRASFORMAZIONI
# considero gender, education, marital come categorici a cui applico OneHotEncoder perchè tanto non c'è una relazione lineare chiara fra l'ordine assegnato e il significato della variabile
#age e income sono numerici e applico una standardizzazione
#considero tutti gli altri dati che sono raccolti su una scala Likert come inrvalli e quindi dati numerici. Voglio normalizzarli usando 0 come media e varianza uno. uso uno Z-score anche qui

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

#provo prima a plottare le distribuzioni e vedere se ci sono outlier
df_categorical = df_2.iloc[:, 1:4]
df_numerical = pd.concat([df_2.iloc[:,0],df_2.iloc[:,4:]], axis=1)

#plt.figure(1)
#df_numerical.hist(figsize=(15, 15), bins=50)
#plt.tight_layout()  # Garantisce che gli istogrammi non si sovrappongano

#ho provato a fare una trasformazione logaritmica ma faceva schifo
# df_log = np.log1p(df_numerical)
# plt.figure(2)
# df_log.hist(figsize=(15, 15), bins=50)
# plt.show()

#provo ad applicare Box-Cox a age ma fa schifo comunque

# from scipy.stats import boxcox
#
# df_age = df_numerical[['age']]  # Nota i doppio parentesi quadre per mantenere un DataFrame
# df_age['age_transformed'], lambda_value = boxcox(df_age['age'])
#
# plt.figure()
# plt.subplot(1, 2, 1)
# df_age['age'].hist()
# plt.title('Distribuzione Originale')
#
# plt.subplot(1, 2, 2)
# df_age['age_transformed'].hist()
# plt.title('Distribuzione dopo Box-Cox')
#
# plt.tight_layout()
# plt.show()

#vediamo se ci sono outlier prima di standardizzare. Al massimo mi posso aspettare outlier solo su age e income

age = df_numerical.iloc[:,0].tolist()
income = df_numerical.iloc[:, 1].tolist()
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

percentiles_age_pre = get_statistics(age)
percentiles_income_pre = get_statistics(income)

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

get_outliers(age, percentiles_age_pre)
get_outliers(income, percentiles_income_pre) #abbiamo come oulier 56000 e 54600 io non farei nulla perchè sono comunqque valori possibili, ce ne sono altri non tanto più bassi. Possiamo decidere poi se toglierli

#print(df_categorical.head(5))
encoder = OneHotEncoder(sparse = False)
df_encoded = pd.DataFrame(encoder.fit_transform(df_categorical))

#print(df_encoded.shape)  16 colonne
#print(df_encoded.head(5))

scaler = StandardScaler()

#print(df_numerical.head(5))
#print(df_numerical.shape)

df_standardized = pd.DataFrame(scaler.fit_transform(df_numerical))
#print(df_standardized.head(5))

#outlier identification dopo la standardizzazione

age_standardized = df_standardized.iloc[:,0].tolist()
income_standardized = df_standardized.iloc[:, 1].tolist()

percentiles_age_post = get_statistics(age_standardized)
percentiles_income_post = get_statistics(income_standardized)

get_outliers(age_standardized, percentiles_age_post)
get_outliers(income_standardized, percentiles_income_post)
#mi rimangono come outlier di income [2.828593068160608, 2.6818167115329334] si mi sa che sarà meglio toglierli

# Concatenating the two dataframes
df_end = pd.concat([df_standardized.iloc[:,0],df_encoded, df_standardized.iloc[:,1:]], axis = 1)
print(df_end.head())  #volevo riassegnare l'ordine delle colonne del dataframe originario ma devo stare attenta che le categoriche sono aumentate in numero

