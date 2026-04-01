"""
COMPLETE COVID-19 DATA ANALYSIS
This script performs comprehensive data analysis including loading, cleaning, 
aggregation, feature engineering, and visualization of COVID-19 data.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0-8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("COMPLETE COVID-19 DATA ANALYSIS")
print("="*80)

# ============================================
# 1. Import the dataset using Pandas
# ============================================
print("\n1. IMPORTING DATASET")
print("-"*50)

url =https://raw.githubusercontent.com/SR1608/Datasets/master/covid-data.csv
df = pd.read_csv(url)
print(f"✓ Dataset loaded successfully")
print(f"  Original shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================
# 2. High Level Data Understanding
# ============================================
print("\n2. HIGH LEVEL DATA UNDERSTANDING")
print("-"*50)

# a. Find no. of rows & columns in the dataset
print(f"\na. Dataset dimensions:")
print(f"   - Number of rows: {df.shape[0]:,}")
print(f"   - Number of columns: {df.shape[1]}")

# b. Data types of columns
print(f"\nb. Data types of columns:")
print(df.dtypes.value_counts())
print(f"\n   Detailed column types:")
for dtype, cols in df.dtypes.groupby(df.dtypes):
    print(f"   {dtype}: {len(cols)} columns")

# c. Info & describe of data in dataframe
print(f"\nc. DataFrame Info:")
buffer = []
df.info(buf=buffer)
info_str = '\n'.join(buffer)
print(info_str)

print(f"\nd. Statistical Summary (numeric columns):")
print(df.describe().round(2))

# ============================================
# 3. Low Level Data Understanding
# ============================================
print("\n3. LOW LEVEL DATA UNDERSTANDING")
print("-"*50)

# a. Find count of unique values in location column
print(f"\na. Unique values in location column: {df['location'].nunique():,}")

# b. Find which continent has maximum frequency using values counts
print(f"\nb. Continent frequency distribution:")
continent_counts = df['continent'].value_counts()
print(continent_counts)
print(f"\n   Continent with maximum frequency: {continent_counts.index[0]} ({continent_counts.values[0]:,} records)")

# c. Find maximum & mean value in 'total_cases'
print(f"\nc. Total Cases statistics:")
print(f"   - Maximum: {df['total_cases'].max():,.0f}")
print(f"   - Mean: {df['total_cases'].mean():,.2f}")
print(f"   - Median: {df['total_cases'].median():,.2f}")

# d. Find 25%,50% & 75% quartile value in 'total_deaths'
print(f"\nd. Total Deaths quartiles:")
print(f"   - 25th percentile (Q1): {df['total_deaths'].quantile(0.25):,.2f}")
print(f"   - 50th percentile (Median): {df['total_deaths'].quantile(0.50):,.2f}")
print(f"   - 75th percentile (Q3): {df['total_deaths'].quantile(0.75):,.2f}")

# e. Find which continent has maximum 'human_development_index'
hdi_by_continent = df.groupby('continent')['human_development_index'].max()
max_hdi_continent = hdi_by_continent.idxmax()
max_hdi_value = hdi_by_continent.max()
print(f"\ne. Continent with maximum Human Development Index:")
print(f"   - Continent: {max_hdi_continent}")
print(f"   - Maximum HDI value: {max_hdi_value:.3f}")

# f. Find which continent has minimum 'gdp_per_capita'
gdp_by_continent = df.groupby('continent')['gdp_per_capita'].min()
min_gdp_continent = gdp_by_continent.idxmin()
min_gdp_value = gdp_by_continent.min()
print(f"\nf. Continent with minimum GDP per capita:")
print(f"   - Continent: {min_gdp_continent}")
print(f"   - Minimum GDP per capita: {min_gdp_value:.2f}")

# ============================================
# 4. Filter the dataframe with required columns
# ============================================
print("\n4. FILTERING DATAFRAME")
print("-"*50)

required_columns = ['continent', 'location', 'date', 'total_cases', 
                    'total_deaths', 'gdp_per_capita', 'human_development_index']
df_filtered = df[required_columns].copy()
print(f"✓ Filtered to {len(required_columns)} columns")
print(f"  New shape: {df_filtered.shape[0]:,} rows × {df_filtered.shape[1]} columns")
print(f"  Columns: {', '.join(df_filtered.columns)}")

# ============================================
# 5. Data Cleaning
# ============================================
print("\n5. DATA CLEANING")
print("-"*50)

# a. Remove all duplicates observations
initial_rows = df_filtered.shape[0]
df_filtered = df_filtered.drop_duplicates()
duplicates_removed = initial_rows - df_filtered.shape[0]
print(f"\na. Duplicate removal:")
print(f"   - Rows before: {initial_rows:,}")
print(f"   - Rows after: {df_filtered.shape[0]:,}")
print(f"   - Duplicates removed: {duplicates_removed:,}")

# b. Find missing values in all columns
print(f"\nb. Missing values before cleaning:")
missing_before = df_filtered.isnull().sum()
print(missing_before[missing_before > 0] if any(missing_before > 0) else "No missing values found")

# c. Remove all observations where continent column value is missing
rows_with_null_continent = df_filtered['continent'].isnull().sum()
df_filtered = df_filtered.dropna(subset=['continent'])
print(f"\nc. Remove rows with missing continent:")
print(f"   - Rows removed: {rows_with_null_continent:,}")
print(f"   - New shape: {df_filtered.shape[0]:,} rows")

# d. Fill all missing values with 0
missing_before_fill = df_filtered.isnull().sum().sum()
df_filtered = df_filtered.fillna(0)
missing_after_fill = df_filtered.isnull().sum().sum()
print(f"\nd. Fill remaining missing values with 0:")
print(f"   - Missing values before: {missing_before_fill}")
print(f"   - Missing values after: {missing_after_fill}")

print(f"\n✓ Data cleaning completed successfully!")

# ============================================
# 6. Date time format and Month extraction
# ============================================
print("\n6. DATE FORMAT CONVERSION")
print("-"*50)

# a. Convert date column in datetime format
print(f"\na. Converting date column to datetime format...")
print(f"   - Before conversion: {df_filtered['date'].dtype}")
df_filtered['date'] = pd.to_datetime(df_filtered['date'])
print(f"   - After conversion: {df_filtered['date'].dtype}")

# b. Create new column month after extracting month data from date column
df_filtered['month'] = df_filtered['date'].dt.month
print(f"\nb. Month column created successfully")
print(f"   - Sample data:")
print(df_filtered[['date', 'month']].head(10))

# ============================================
# 7. Data Aggregation
# ============================================
print("\n7. DATA AGGREGATION")
print("-"*50)

# a. Find max value in all columns using groupby function on 'continent' column
# b. Store the result in a new dataframe named 'df_groupby'
df_groupby = df_filtered.groupby('continent').max().reset_index()
print(f"✓ Groupby aggregation completed")
print(f"  df_groupby shape: {df_groupby.shape[0]} rows × {df_groupby.shape[1]} columns")
print(f"\n  Aggregated data by continent:")
print(df_groupby[['continent', 'total_cases', 'total_deaths', 'gdp_per_capita', 
                   'human_development_index']].round(2))

# ============================================
# 8. Feature Engineering
# ============================================
print("\n8. FEATURE ENGINEERING")
print("-"*50)

# a. Create a new feature 'total_deaths_to_total_cases' by ratio
# Handle division by zero cases
df_groupby['total_deaths_to_total_cases'] = df_groupby.apply(
    lambda row: row['total_deaths'] / row['total_cases'] if row['total_cases'] > 0 else 0, 
    axis=1
)
print(f"✓ New feature 'total_deaths_to_total_cases' created")
print(f"\n  Death to Cases Ratio by Continent:")
ratio_data = df_groupby[['continent', 'total_cases', 'total_deaths', 'total_deaths_to_total_cases']].copy()
ratio_data['total_deaths_to_total_cases'] = ratio_data['total_deaths_to_total_cases'].apply(lambda x: f"{x:.2%}")
print(ratio_data.to_string(index=False))

# ============================================
# 9. Data Visualization
# ============================================
print("\n9. DATA VISUALIZATION")
print("-"*50)
print("Generating visualizations...\n")

# a. Univariate analysis on 'gdp_per_capita' column using histogram
print("a. Histogram of GDP per capita")
plt.figure(figsize=(10, 6))
sns.histplot(df_groupby['gdp_per_capita'], bins=20, kde=True, color='steelblue', edgecolor='black')
plt.title('Distribution of GDP per Capita by Continent', fontsize=14, fontweight='bold')
plt.xlabel('GDP per Capita (USD)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# b. Scatter plot of 'total_cases' & 'gdp_per_capita'
print("\nb. Scatter plot: Total Cases vs GDP per Capita")
plt.figure(figsize=(12, 7))
scatter = sns.scatterplot(data=df_groupby, x='gdp_per_capita', y='total_cases', 
                          hue='continent', size='total_cases', sizes=(50, 400),
                          alpha=0.7, edgecolor='black', linewidth=1.5)
plt.title('Relationship between Total COVID-19 Cases and GDP per Capita', 
          fontsize=14, fontweight='bold')
plt.xlabel('GDP per Capita (USD)', fontsize=12)
plt.ylabel('Total Cases', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Continent')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# c. Plot Pairplot on df_groupby dataset
print("\nc. Pairplot of df_groupby dataset")
numeric_columns = ['total_cases', 'total_deaths', 'gdp_per_capita', 
                   'human_development_index', 'total_deaths_to_total_cases', 'month']
df_pairplot = df_groupby[numeric_columns + ['continent']].copy()

pairplot = sns.pairplot(df_pairplot, hue='continent', diag_kind='kde', 
                        plot_kws={'alpha': 0.6, 's': 60, 'edgecolor': 'black'},
                        diag_kws={'fill': True}, height=2.5)
pairplot.fig.suptitle('Pairplot Analysis of COVID-19 Metrics by Continent', 
                      y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# d. Bar plot of 'continent' column with 'total_cases'
print("\nd. Bar plot: Total Cases by Continent")
plt.figure(figsize=(12, 6))
bars = sns.barplot(data=df_groupby, x='continent', y='total_cases', 
                   palette='viridis', errorbar=None)
plt.title('Total COVID-19 Cases by Continent', fontsize=14, fontweight='bold')
plt.xlabel('Continent', fontsize=12)
plt.ylabel('Total Cases', fontsize=12)
plt.xticks(rotation=45)

# Add value labels on top of bars
for i, bar in enumerate(bars.patches):
    height = bar.get_height()
    bars.text(bar.get_x() + bar.get_width()/2., height,
              f'{height:,.0f}', ha='center', va='bottom', fontsize=10, rotation=0)
plt.tight_layout()
plt.show()

# Additional Visualization 1: Bar plot of Human Development Index
print("\ne. Additional: Human Development Index by Continent")
plt.figure(figsize=(12, 6))
bars2 = sns.barplot(data=df_groupby, x='continent', y='human_development_index', 
                    palette='rocket', errorbar=None)
plt.title('Human Development Index by Continent', fontsize=14, fontweight='bold')
plt.xlabel('Continent', fontsize=12)
plt.ylabel('Human Development Index', fontsize=12)
plt.xticks(rotation=45)
plt.ylim(0, 1)

for i, bar in enumerate(bars2.patches):
    height = bar.get_height()
    bars2.text(bar.get_x() + bar.get_width()/2., height,
              f'{height:.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()

# Additional Visualization 2: Monthly trend analysis
print("\nf. Additional: Monthly Total Cases Trend by Continent")
monthly_trend = df_filtered.groupby(['continent', 'month'])['total_cases'].max().reset_index()

plt.figure(figsize=(14, 8))
for continent in monthly_trend['continent'].unique():
    continent_data = monthly_trend[monthly_trend['continent'] == continent]
    plt.plot(continent_data['month'], continent_data['total_cases'], 
             marker='o', linewidth=2, markersize=8, label=continent)

plt.title('Monthly Total Cases Trend by Continent', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Cases', fontsize=12)
plt.legend(title='Continent', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.show()

# Additional Visualization 3: Death to Cases Ratio
print("\ng. Additional: Death to Cases Ratio by Continent")
plt.figure(figsize=(12, 6))
bars3 = sns.barplot(data=df_groupby, x='continent', y='total_deaths_to_total_cases', 
                    palette='coolwarm', errorbar=None)
plt.title('Death to Cases Ratio by Continent', fontsize=14, fontweight='bold')
plt.xlabel('Continent', fontsize=12)
plt.ylabel('Death to Cases Ratio', fontsize=12)
plt.xticks(rotation=45)

for i, bar in enumerate(bars3.patches):
    height = bar.get_height()
    bars3.text(bar.get_x() + bar.get_width()/2., height,
              f'{height:.2%}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()

# ============================================
# 10. Save the df_groupby dataframe
# ============================================
print("\n10. SAVING FINAL DATAFRAMES")
print("-"*50)

# Save main aggregated dataframe
df_groupby.to_csv('df_groupby_covid_analysis.csv', index=False)
print(f"✓ df_groupby saved as 'df_groupby_covid_analysis.csv'")

# Save cleaned filtered data
df_filtered.to_csv('cleaned_covid_data_complete.csv', index=False)
print(f"✓ Cleaned data saved as 'cleaned_covid_data_complete.csv'")

# Save monthly trend data
monthly_trend.to_csv('monthly_trend_covid.csv', index=False)
print(f"✓ Monthly trend data saved as 'monthly_trend_covid.csv'")

# ============================================
# Final Summary Report
# ============================================
print("\n" + "="*80)
print("FINAL SUMMARY REPORT")
print("="*80)

print("\n📊 DATASET OVERVIEW:")
print(f"   - Original dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   - Cleaned dataset: {df_filtered.shape[0]:,} rows × {df_filtered.shape[1]} columns")
print(f"   - Aggregated dataset: {df_groupby.shape[0]} rows × {df_groupby.shape[1]} columns")

print("\n📈 KEY METRICS BY CONTINENT:")
print("-"*60)
print(f"{'Continent':<15} {'Total Cases':<15} {'GDP per Capita':<15} {'HDI':<10} {'Death Ratio':<12}")
print("-"*60)
for _, row in df_groupby.iterrows():
    print(f"{row['continent']:<15} {row['total_cases']:>12,.0f}  {row['gdp_per_capita']:>12,.0f}  "
          f"{row['human_development_index']:>8.3f}  {row['total_deaths_to_total_cases']:>10.2%}")

print("\n📁 FILES GENERATED:")
print("   1. df_groupby_covid_analysis.csv - Aggregated data by continent")
print("   2. cleaned_covid_data_complete.csv - Complete cleaned dataset")
print("   3. monthly_trend_covid.csv - Monthly trend data for analysis")

print("\n🎯 INSIGHTS SUMMARY:")
print("   • Continent with most records: " + continent_counts.index[0])
print("   • Continent with highest HDI: " + max_hdi_continent)
print("   • Continent with lowest GDP: " + min_gdp_continent)
print(f"   • Overall death to cases ratio ranges from "
      f"{df_groupby['total_deaths_to_total_cases'].min():.2%} to "
      f"{df_groupby['total_deaths_to_total_cases'].max():.2%}")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*80)