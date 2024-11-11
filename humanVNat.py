import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load and preprocess data
file_path = '/Users/rheajohnson/Downloads/caFirePerimeters.csv'
df = pd.read_csv(file_path)

# Map causes
cause_mapping = {
    1: 'Lightning', 2: 'Equipment Use', 3: 'Smoking', 4: 'Campfire', 5: 'Debris',
    6: 'Railroad', 7: 'Arson', 8: 'Playing with fire', 9: 'Miscellaneous', 10: 'Vehicle',
    11: 'Powerline', 12: 'Firefighter Training', 13: 'Non-Firefighter Training',
    14: 'Unknown/Unidentified', 15: 'Structure', 16: 'Aircraft', 17: 'Volcanic',
    18: 'Escaped Prescribed Burn', 19: 'Illegal Alien Campfire'
}
df['Cause'] = df['Cause'].map(cause_mapping)

# Define human and nature causes
human_causes = [
    'Smoking', 'Campfire', 'Equipment Use', 'Playing with fire', 'Arson', 'Railroad',
    'Vehicle', 'Powerline', 'Firefighter Training', 'Non-Firefighter Training',
    'Structure', 'Aircraft', 'Escaped Prescribed Burn', 'Illegal Alien Campfire'
]
nature_causes = ['Lightning', 'Debris']

# Create cause type column
df['Cause_Type'] = df['Cause'].apply(lambda x: 'Human' if x in human_causes else 'Natural' if x in nature_causes else 'Unknown')

# Clean acres data
df['GIS Calculated Acres'] = pd.to_numeric(df['GIS Calculated Acres'], errors='coerce')

# Create multiple visualizations
plt.figure(figsize=(15, 12))

# 1. Box plot comparison (log scale)
plt.subplot(2, 2, 1)
sns.boxplot(x='Cause_Type', y='GIS Calculated Acres', data=df[df['Cause_Type'].isin(['Human', 'Natural'])])
plt.yscale('log')
plt.title('Distribution of Fire Sizes by Cause Type (Log Scale)')
plt.xlabel('Cause Type')
plt.ylabel('Acres (Log Scale)')

# 2. Violin plot comparison
plt.subplot(2, 2, 2)
sns.violinplot(x='Cause_Type', y='GIS Calculated Acres', data=df[df['Cause_Type'].isin(['Human', 'Natural'])])
plt.yscale('log')
plt.title('Distribution of Fire Sizes by Cause Type (Violin Plot)')
plt.xlabel('Cause Type')
plt.ylabel('Acres (Log Scale)')

# 3. Bar plot of mean fire sizes
plt.subplot(2, 2, 3)
cause_type_means = df[df['Cause_Type'].isin(['Human', 'Natural'])].groupby('Cause_Type')['GIS Calculated Acres'].mean()
cause_type_means.plot(kind='bar')
plt.title('Mean Fire Size by Cause Type')
plt.xlabel('Cause Type')
plt.ylabel('Mean Acres')

# 4. Count and proportion plot
plt.subplot(2, 2, 4)
df_large = df.copy()
size_threshold = df['GIS Calculated Acres'].quantile(0.90)
df_large['Size_Category'] = df_large['GIS Calculated Acres'].apply(lambda x: 'Large' if x > size_threshold else 'Small')
cause_size_props = pd.crosstab(df_large['Cause_Type'], df_large['Size_Category'], normalize='index') * 100
cause_size_props['Large'].plot(kind='bar')
plt.title('Proportion of Large Fires by Cause Type')
plt.xlabel('Cause Type')
plt.ylabel('Percentage of Large Fires')

plt.tight_layout()
plt.show()

# Statistical Analysis
print("\nSummary Statistics by Cause Type:")
summary_stats = df[df['Cause_Type'].isin(['Human', 'Natural'])].groupby('Cause_Type')['GIS Calculated Acres'].agg([
    'count',
    'mean',
    'median',
    'std',
    lambda x: x.quantile(0.75),
    'max'
]).round(2)
summary_stats.columns = ['Count', 'Mean Acres', 'Median Acres', 'Std Dev', '75th Percentile', 'Max Acres']
print(summary_stats)

# Mann-Whitney U test
human_sizes = df[df['Cause_Type'] == 'Human']['GIS Calculated Acres'].dropna()
natural_sizes = df[df['Cause_Type'] == 'Natural']['GIS Calculated Acres'].dropna()
statistic, p_value = stats.mannwhitneyu(human_sizes, natural_sizes, alternative='two-sided')

print("\nStatistical Test Results:")
print(f"Mann-Whitney U test p-value: {p_value:.4f}")
if p_value < 0.05:
    print("There is a statistically significant difference in fire sizes between human and natural causes.")
else:
    print("There is no statistically significant difference in fire sizes between human and natural causes.")

# Additional metrics
print("\nProportion of Large Fires:")
for cause_type in ['Human', 'Natural']:
    total_fires = len(df[df['Cause_Type'] == cause_type])
    large_fires = len(df[(df['Cause_Type'] == cause_type) & (df['GIS Calculated Acres'] > size_threshold)])
    print(f"{cause_type}: {(large_fires/total_fires)*100:.1f}% ({large_fires} out of {total_fires})")