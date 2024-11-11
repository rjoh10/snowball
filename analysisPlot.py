import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# Load and preprocess data (using your existing preprocessing steps)
file_path = '/Users/rheajohnson/Downloads/caFirePerimeters.csv'
df = pd.read_csv(file_path)

# Map causes to categories (using your existing mapping)
cause_mapping = {
    1: 'Lightning', 2: 'Equipment Use', 3: 'Smoking', 4: 'Campfire', 5: 'Debris',
    6: 'Railroad', 7: 'Arson', 8: 'Playing with fire', 9: 'Miscellaneous', 10: 'Vehicle Accident',
    11: 'Powerline', 12: 'Firefighter Training', 13: 'Non-Firefighter Training',
    14: 'Unknown/Unidentified', 15: 'Structure', 16: 'Aircraft', 17: 'Volcanic',
    18: 'Escaped Prescribed Burn', 19: 'Illegal Alien Campfire'
}
df['Cause'] = df['Cause'].map(cause_mapping)

# Clean acres data
df['GIS Calculated Acres'] = pd.to_numeric(df['GIS Calculated Acres'], errors='coerce')

# Calculate summary statistics for each cause
cause_stats = df.groupby('Cause')['GIS Calculated Acres'].agg([
    'count',
    'mean',
    'median',
    'std',
    lambda x: x.quantile(0.75),
    'max'
]).round(2)
cause_stats.columns = ['Count', 'Mean Acres', 'Median Acres', 'Std Dev', '75th Percentile', 'Max Acres']
cause_stats = cause_stats.sort_values('Mean Acres', ascending=False)

print("\nSummary Statistics by Cause:")
print(cause_stats)

# Create visualizations
plt.figure(figsize=(15, 8))

# Box plot (log scale for better visualization)
plt.subplot(2, 1, 1)
sns.boxplot(x='Cause', y='GIS Calculated Acres', data=df)
plt.yscale('log')
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Fire Sizes by Cause (Log Scale)')
plt.xlabel('Cause')
plt.ylabel('Acres (Log Scale)')

# Bar plot of mean fire size by cause
plt.subplot(2, 1, 2)
sns.barplot(x='Cause', y='GIS Calculated Acres', data=df, estimator=np.mean, ci=95)
plt.xticks(rotation=45, ha='right')
plt.title('Mean Fire Size by Cause')
plt.xlabel('Cause')
plt.ylabel('Mean Acres')

plt.tight_layout()
plt.show()

# Calculate percentage of large fires by cause
# Define large fires as those above 90th percentile
size_threshold = df['GIS Calculated Acres'].quantile(0.90)
df['is_large'] = df['GIS Calculated Acres'] > size_threshold

large_fire_pct = df.groupby('Cause')['is_large'].agg(['count', 'mean']).round(3)
large_fire_pct.columns = ['Total Fires', 'Proportion Large Fires']
large_fire_pct = large_fire_pct.sort_values('Proportion Large Fires', ascending=False)

print("\nProportion of Large Fires by Cause:")
print(large_fire_pct)

# Perform Kruskal-Wallis H-test to check if fire sizes differ significantly between causes
h_statistic, p_value = stats.kruskal(*[group['GIS Calculated Acres'].dropna()
                                     for name, group in df.groupby('Cause')])

print("\nStatistical Test Results:")
print(f"Kruskal-Wallis H-test:")
print(f"H-statistic: {h_statistic:.2f}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("There is a statistically significant difference in fire sizes between different causes.")
else:
    print("There is no statistically significant difference in fire sizes between different causes.")