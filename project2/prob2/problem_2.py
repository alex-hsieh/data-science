import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# URL to source data
url = "https://paleobiodb.org/data1.2/occs/list.csv?base_name=Dinosauria&min_ma=65&max_ma=200&vocab=pbdb&show=classext,ident"

# Fetch the data from the URL
df = pd.read_csv(url)
print(df.columns.tolist())


# Confirm it worked
# print(f"Successfully fetched {df.shape[0]} records!")
# print(df.head())

# Data cleaning and filtering
genus_df = df[df['accepted_rank'] == 'genus']
genus_df = genus_df.dropna(subset=['genus'])

# Feature engineering of the summary table
summary_df = genus_df.groupby('genus').agg({
    'max_ma': 'max',
    'min_ma': 'min',
    'occurrence_no': 'count',
})

# Add a new column to the summary dataframe that calculates the duration 
# of each genus (max_ma - min_ma)
summary_df['temporal_range'] = summary_df['max_ma'] - summary_df['min_ma']

# Scaling the features using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(summary_df[['max_ma', 
                                                   'min_ma', 
                                                   'occurrence_no', 
                                                   'temporal_range']])
scaled_df = pd.DataFrame(scaled_features, columns=['max_ma_scaled', 
                                                   'min_ma_scaled', 
                                                   'occurrence_no_scaled', 
                                                   'temporal_range_scaled'], 
                                                   index=summary_df.index)

# computere the linkage matrix for hierarchical clustering
Z = linkage(scaled_df, method='complete')

# Plot the dendrogram
plt.figure(figsize=(15, 8))
dendrogram(Z, labels=scaled_df.index.tolist(), leaf_rotation=90, leaf_font_size=8)
plt.title('Dinosaur Genus Hierarchical Clustering Dendrogram (Complete Linkage)')
plt.xlabel('Genus')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()