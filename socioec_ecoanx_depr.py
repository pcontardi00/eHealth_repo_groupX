
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



# cerco di capire il numero migliore di componenti per applicare la PCA

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(df_end)
varianza_spiegata = pca.explained_variance_ratio_

plt.plot(np.cumsum(varianza_spiegata))
plt.xlabel('Numero di componenti')
plt.ylabel('Varianza spiegata cumulativa')
#plt.show()

# utilizzo il metodo di Kaiser per determinare il numero di componenti principali desiderate per applicare la PCA
# pca = PCA()
# pca.fit(df_end)
#
# # Estrai gli autovalori
# autovalori = pca.singular_values_**2 / (df_end.shape[0] - 1)
#
# # Applica il criterio di Kaiser
# n_components_kaiser = sum(autovalori > 1)
# print(f"Numero di componenti da conservare secondo il criterio di Kaiser: {n_components_kaiser}")

#mi dice che il numero di componenti è 8

# Applicare PCA
n_components = 8  # Numero di componenti principali desiderate
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(df_end)

print("Varianza spiegata da ogni componente:", pca.explained_variance_ratio_)
#print("Dati proiettati sui primi 2 componenti principali:\n", X_pca)

# a me sembra che venga spiegata troppa poca varianza

#decido si non fare il bilanciamento del dataset

# cerchiamo il numero ottimale di clusters

from sklearn.cluster import KMeans

distortions = []
K = range(1, 50)  # Cambia in base al tuo dataset e alle tue esigenze
for k in K:
    kmeanModel = KMeans(n_clusters=k, n_init=10)
    kmeanModel.fit(X_pca)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16, 8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Numero di cluster')
plt.ylabel('Distorsione')
plt.title('Metodo del gomito mostrando il numero ottimale di cluster')
#plt.show()
#mi sembra 5

# Applicare KMeans
kmeans = KMeans(n_clusters=8, n_init=10, random_state=77)   # S= 0.2889 con 8 clusters con 9 diminuisce
clusters = kmeans.fit_predict(X_pca)

# Visualizzazione dei cluster
plt.figure(3)
plt.scatter(X_pca[:, 1], X_pca[:, 2], c=clusters, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroidi')
plt.legend()
plt.title("KMeans Clustering")
#plt.show()
#fa proprio un po' schifo

#provo a valutalo con una metrica
# Silhouette Coefficient: Valori vicini a 1 indicano una buona clusterizzazione, mentre valori vicini a -1 indicano una cattiva clusterizzazione.
# Calinski-Harabasz Index: Valori più alti indicano cluster densi e ben separati.
# Davies-Bouldin Index: Valori più bassi indicano cluster meglio definiti.

from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids

silhouette_score = metrics.silhouette_score(X_pca, clusters)
calinski_harabasz_score = metrics.calinski_harabasz_score(X_pca, clusters)
davies_bouldin_score = metrics.davies_bouldin_score(X_pca, clusters)

print("silhouette means:"+str(silhouette_score))
print("calinski_harabasz_score means"+str(calinski_harabasz_score))
print("davies_bouldin_score means" + str(davies_bouldin_score))

# valuto il numero di clusters ottimale per k-medoids

def compute_distortion(data, kmedoids_model):
    cluster_centers = kmedoids_model.cluster_centers_
    labels = kmedoids_model.labels_
    total_distortion = 0

    for cluster_idx, medoid in enumerate(cluster_centers):
        members_idx = np.where(labels == cluster_idx)[0]
        members = data[members_idx]
        distances = pairwise_distances(members, medoid.reshape(1, -1))
        total_distortion += np.sum(distances)

    return total_distortion

distortions_med = []
silhouettes_med = []
K_range_med = range(2, 7)  # Supponendo che desideri testare tra 2 e 15 cluster

for K in K_range_med:
    kmedoids = KMedoids(n_clusters=K, random_state=0)
    kmedoids.fit(X_pca)
    clusters_prova = kmedoids.predict(X_pca)

    distortions_med.append(compute_distortion(X_pca, kmedoids))
    # silhouette_avg = silhouette_score(X_pca, clusters_prova)
    # silhouettes_med.append(silhouette_avg)

# Metodo del gomito
plt.figure(5)
plt.plot(K_range_med, distortions_med, 'bx-')
plt.xlabel('Numero di cluster')
plt.ylabel('Distorsione')
plt.title('Metodo del gomito per K-medoids')
#plt.show()

# applico k-medoids

kmedoids = KMedoids(n_clusters=3, random_state=77)
clusters_med = kmedoids.fit_predict(X_pca)

# Visualizzazione dei cluster
plt.figure(4)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_med, s=50, cmap='viridis')
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Medoids')
plt.legend()
plt.title("KMedoids Clustering")
plt.show()

#valuto K-medoids con metriche

silhouette_score_med = metrics.silhouette_score(X_pca, clusters_med)
calinski_harabasz_score_med = metrics.calinski_harabasz_score(X_pca, clusters_med)
davies_bouldin_score_med = metrics.davies_bouldin_score(X_pca, clusters_med)

print("silhouette med: "+str(silhouette_score_med)) #0.06 con 4 che mi sembra più giusto dall'elbow    0.14 con 3
print("calinski_harabasz_score med: "+str(calinski_harabasz_score_med))
print("davies_bouldin_score med: " + str(davies_bouldin_score_med))
