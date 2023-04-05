# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:20:08 2023

@author: Vipin Vinod
"""

# Function to read the name of the dataset and return two datasets


from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def read_data(data):
    # Reading the data
    try:
        df = pd.read_csv(
            r'C:\Users\Vipin Vinod\Desktop\assignment\ads2\%s' %
            data, on_bad_lines='skip')
    except pd.errors.ParserError:
        print(f"Error parsing file {data}")
        return None

    # Transpose the data and name the new column as 'Years'
    t_df = df.transpose().reset_index()
    t_df.columns = t_df.iloc[0]
    t_df = t_df[1:]

    return df, t_df


df, t_df = read_data('climate_change.csv')  # Getting the two datasets
df.head()

# Checking all the available indicators
df['Indicator Name'].unique()

# Transposed DataFrame
t_df.head()

# creating 3 dataframes with 3 indicators by subsetting the original dataframe
df_renew = df[df['Indicator Name'] ==
              'Renewable energy consumption (% of total final energy consumption)']
df_accelec = df[df['Indicator Name'] ==
                'Access to electricity (% of population)']
df_popgrow = df[df['Indicator Name'] == 'Population growth (annual %)']


# merging the 3 dataframes
df_merge = pd.concat([df_renew, df_accelec, df_popgrow])

# Resetting the index
df_merge.reset_index(inplace=True, drop=True)

# replace '..' values with NaN values in data
df_merge.replace('..', np.nan, inplace=True)
# removing the past 4 years since there is very little data
df_merge = df_merge.iloc[:, :-3]

# replacing missing values with 0
df_merge.fillna('0', inplace=True)  # replacing NaN with 0's
# dropping redundant columns
df_merge.drop(['Indicator Code', 'Country Code'], axis=1, inplace=True)

years = df_merge.columns[2:]
# removing wrong values final time
df_merge.replace('..', 0, inplace=True)

# Converting the data to numeric and rounding off to 2 decimals
df_merge[years] = np.abs(df_merge[years].astype('float').round(decimals=2))


# Checking of the size of the 3 datasets
df_merge.groupby(['Indicator Name'])['1960'].size().plot(kind='bar')
plt.title('Size of each group')
plt.xticks(rotation=90)
plt.show()

# Plotting the means of the 3 groups of the data over the years
figure, axes = plt.subplots(1, 2)

df_merge.groupby(['Indicator Name'])['1990'].mean().plot(
    kind='bar', ax=axes[0], color='turquoise')
axes[0].title.set_text('Mean of each group in the year 1961')

df_merge.groupby(['Indicator Name'])['2018'].mean().plot(
    kind='bar', ax=axes[1], color='red')
axes[1].title.set_text('Mean of each group in the year 2018')
# Add gap between subplots
plt.subplots_adjust(wspace=0.8)
plt.savefig('change.jpeg')
plt.show()


# Using describe() to get the summary statistics of the merged dataframe
print(df_merge.loc[:, '2000':'2018'].describe())


# Select only numeric columns
numeric_cols = df_merge.select_dtypes(include=np.number).columns.tolist()

# Compute skewness and kurtosis for each numeric column
for col in numeric_cols:
    skewness = skew(df_merge[col])
    kurt = kurtosis(df_merge[col])
    print(f"Column {col}: Skewness = {skewness:.2f}, Kurtosis = {kurt:.2f}")


# diving the data back to 3 datasets to perform summary statistics
df_merge_renew = df_merge[df_merge['Indicator Name'] ==
                          'Renewable energy consumption (% of total final energy consumption)']
df_merge_accelec = df_merge[df_merge['Indicator Name']
                            == 'Access to electricity (% of population)']
df_merge_popgrow = df_merge[df_merge['Indicator Name']
                            == 'Population growth (annual %)']


# saving the summary statistics in a dataframe
stats = pd.DataFrame()
stats['renew'] = df_merge_renew[years].mean(axis=0).to_frame()
stats['accelec'] = df_merge_accelec[years].mean(axis=0).to_frame()
stats['popgrow'] = df_merge_popgrow[years].mean(axis=0).to_frame()

# select columns for the years 2000 to 2018
stats = stats.loc['2000':'2018']
# scaling the data since all the groups are not in the same scale
scaler = StandardScaler()

# scaling the data using standard scaler and saving it with full column names
stats_sc = pd.DataFrame(
    scaler.fit_transform(stats),
    columns=[
        'Renewable energy consumption',
        'Access to electricity (% of population)',
        'Population growth (annual %)'])

stats_sc.index = stats.index  # setting the index

# plotting the general trend of the 3 groups over the years

plt.figure(figsize=(6, 6))  # size of the image
for i in stats_sc.columns:
    plt.plot(stats_sc.index, stats_sc[i], label=i)  # plotting the 3 lines
plt.legend(loc='best')  # setting the location of the legends
plt.xticks(stats_sc.index, rotation=45)  # changing the x axis values
plt.ylabel('Relative Change')  # setting y axis label
# setting the title
plt.title('Change of Indicators over the years all over the world')

plt.savefig('trend.jpeg')  # saving the image
plt.show()

Haiti = df_merge[df_merge['Country Name'] == 'Haiti']
Haiti = Haiti.drop(['Country Name'], axis=1)
Haiti = Haiti.set_index('Indicator Name').loc[:, '2000':'2018'].T

Greece = df_merge[df_merge['Country Name'] == 'Greece']
Greece = Greece.drop(['Country Name'], axis=1)
Greece = Greece.set_index('Indicator Name').loc[:, '2000':'2018'].T


def plot_country(data, name):
    plt.figure(figsize=(6, 6))
    for i in data.columns:
        # Select only the data for the years 2000 to 2018
        data_range = data.loc['2000':'2018', [i]]

        scaler = StandardScaler()
        # scaling the data using standard scaler and saving it with full column
        # names
        country_sc = pd.DataFrame(
            scaler.fit_transform(data_range), columns=[i])
        country_sc.index = data_range.index
        plt.plot(
            country_sc.index,
            country_sc[i],
            label=i)  # plotting the line
        plt.legend(loc='best')  # setting the location of the legends
        plt.xticks(country_sc.index, rotation=45)  # changing the x axis values
        plt.ylabel('Relative Change')  # setting y axis label
    plt.title(
        'Change of Indicators over the years in %s' %
        name)  # setting the title
    plt.show()


plot_country(Haiti, 'Haiti')
plot_country(Greece, 'Greece')


corr_mat = stats_sc.corr()  # calculating the correlation of the data
fig, ax = plt.subplots(figsize=(10, 10))  # size of the figure
im = ax.imshow(corr_mat, cmap='plasma')  # plotting the heatmap

# adding the values inside each cell
for i in range(len(corr_mat)):
    for j in range(len(corr_mat)):
        text = ax.text(j, i, round(
            corr_mat.iloc[i, j], 2), ha="center", va="center", color="black")

# setting the labels
ax.set_xticks(np.arange(len(corr_mat.columns)))
ax.set_yticks(np.arange(len(corr_mat.columns)))
ax.set_xticklabels(corr_mat.columns)
ax.set_yticklabels(corr_mat.columns)

# rotating the x axis labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# adding the color bar
cbar = ax.figure.colorbar(im, ax=ax)

# setting the title
ax.set_title("Correlation Heatmap")

# saving and showing the image
plt.savefig('correlation_heatmap.png')
plt.show()
