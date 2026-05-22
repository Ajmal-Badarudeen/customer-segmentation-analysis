
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


df = pd.read_csv('D:/Projects/Customer segmentation/customers.csv')

X = df[['Annual_Income', 'Spending_Score']]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)

df['Cluster'] = kmeans.fit_predict(X_scaled)


print(df.groupby('Cluster')[['Age', 'Annual_Income', 'Spending_Score']].mean())


plt.figure(figsize=(8,6))
plt.scatter(
    df['Annual_Income'],
    df['Spending_Score'],
    c=df['Cluster']
)

plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.legend()

plt.show()

df.to_csv(
    "D:/Projects/Customer segmentation/customers_with_clusters.csv",
    index=False
)
