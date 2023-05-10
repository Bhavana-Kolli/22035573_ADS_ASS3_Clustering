#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7PAM2000 Applied Data Science 1
Assignment 3: Clustering and fitting

@author: Bhavana Kolli - 22035573
"""

# Here modules are imported

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn import cluster for clustering
import sklearn.cluster as cluster
import sklearn.metrics as skmet

# to do fitting
import scipy.optimize as opt

# import modules provided
import cluster_tools as ct
import errors as err


# Here functions are defined


def read_df(filename):
    """
    Reads a dataframe in World Bank format from a CSV file.
    Transposes and cleans the dataframe.
    
    Args:
        filename (str): the name of the CSV file to be read
    Returns:
        df_years(pd.DataFrame): 
            a dataframe which is cleaned and transposed
    """

    # read the CSV file
    df = pd.read_csv(filename, skiprows=4)

    # drops unnecessary columns
    df.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1,
            inplace=True)
    
    # sets the index to the country name
    df = df.set_index('Country Name')

    # drops rows and columns with all missing data
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # transpose the dataframe
    df_t = df.transpose()

    # convert the index to years
    df_t.index = pd.to_datetime(df_t.index, format='%Y').year

    # a dataframe countries as columns
    df_countries = df_t

    # a dataframe with years as columns
    df_years = df_countries.T
    
    # resets the index
    df_years = df_years.reset_index()
    
    return df_years


def get_year_data_merge(df1, df2, year1, year2):
    """
    Extracts a single year column from a given dataframe.
    
    Args:
        df (pandas.DataFrame): The dataframe to extract data from.
        year (int): The year to extract data for.
    
    Returns:
        pandas.DataFrame: A dataframe containing the data for the given year.
    """
    
    # drop rows with nan's in the given year
    df1 = df1.dropna(subset=[year1, year2])
    df2 = df2.dropna(subset=[year1, year2])
    
    df1_year = df1[["Country Name", year1, year2]].copy()
    df2_year = df2[["Country Name", year1, year2]].copy()
    
    df_year = pd.merge(df1_year, df2_year, on="Country Name", how="outer")
    
    df_year = df_year.dropna() 
    
    return df_year


def plot_correlation_heatmaps(df1, df2, title1, title2, size=6):
    """
    Plots correlation heatmaps side by side for two dataframes with 
    two different scales.

    Args:
        df1, df2 (pandas DataFrame): Input dataframes
        title1, title2 (str): Titles for the heatmaps
        size (int): The vertical and horizontal size of the plot (in inches)
    """
    fig, axes = plt.subplots(1, 2, figsize=(2*size, size))

    # Plot heatmap for df1 with coolwarm color scale
    corr1 = df1.corr()
    im1 = axes[0].matshow(corr1, cmap='coolwarm')
    axes[0].set_xticks(range(len(corr1.columns)), fontsize=15)
    axes[0].set_yticks(range(len(corr1.columns)), fontsize=15)
    axes[0].set_xticklabels(corr1.columns, rotation=90, fontsize=15)
    axes[0].set_yticklabels(corr1.columns, fontsize=15)
    axes[0].set_title(title1, fontsize=15)
    fig.colorbar(im1, ax=axes[0])

    # Plot heatmap for df2 with coolwarm color scale
    corr2 = df2.corr()
    im2 = axes[1].matshow(corr2, cmap='coolwarm')
    axes[1].set_xticks(range(len(corr2.columns)), fontsize=15)
    axes[1].set_yticks(range(len(corr2.columns)), fontsize=15)
    axes[1].set_xticklabels(corr2.columns, rotation=90, fontsize=15)
    axes[1].set_yticklabels(corr2.columns, fontsize=15)
    axes[1].set_title(title2, fontsize=15)
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()


def fit_clusters(df, n_clusters):
    """
    Fits the data to a K-means clustering model 
    and returns the labels and centroids.

    Args:
        df (pandas DataFrame): Input data for clustering
        n_clusters (int): Number of clusters to create

    Returns:
        labels (array-like): Cluster labels for each data point
        centroids (array-like): Coordinates of the cluster centroids
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df)
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    return labels, cen



def plot_clusters(df, labels, cen, x_label, y_label, title):
    """
    Plots a scatter plot of data points colored by clusters 
    and displays the centroids.

    Args:
        df (pandas DataFrame): Input data for clustering
        labels (array-like): Cluster labels for each data point
        cen (array-like): Coordinates of the cluster centres
        x_label (str): Column name for x-axis
        y_label (str): Column name for y-axis
        title (str): Title of cluster plot
    """
    
    # Extract the estimated cluster centres for x, y axis
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    
    # Plot
    plt.figure(figsize=(12.0, 8.0))
    cm = plt.cm.get_cmap('viridis')
    scatter = plt.scatter(df[x_label], df[y_label], 30, 
                          labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 90, "r", marker="d")
    
    # add x, y labels and title
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.title(title, fontsize=20)
    
    # Add legend for clusters and centroids at top corner
    plt.legend(handles=scatter.legend_elements()[0] + 
               [plt.scatter([], [], marker='D', color='r')],
               labels=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Centroids'],
               fontsize=15, loc='upper left')
    
    # show the plot
    #plt.show()




# Main Program


# Reading Files-------------------------------------------------------

# read the data for "CO2 emissions (metric tons per capita)"
df_co2 = read_df("co2 emissions.csv")

# read the data for "GDP per capita (current US$)"
df_gdp = read_df("gdp per capita.csv")


# Summary Statistics--------------------------------------------------


# summary statistics for "CO2 emissions(metric tons per capita)" of whole world
print("\nCO2 emissions summary statistics for whole world:")
print(df_co2.describe())

# summary statistics for "GDP per capita (current US$)" of whole world
print("\nGDP per capita summary statistics for whole world:")
print(df_gdp.describe())

df_co2_y = df_co2[[1990, 2000, 2010, 2019]]

df_gdp_y = df_gdp[[1990, 2000, 2010, 2019]]

# Plot correlation heatmaps for CO2 and GDP
plot_correlation_heatmaps(df_co2_y, df_gdp_y, 
                          "CO2 Emissions", "GDP per capita")

# find data of co2 emissions & gdp per capita for the year 1990 and 2019
df_1990_2019 = get_year_data_merge(df_co2, df_gdp, 1990, 2019)

# rename columns
df_1990_2019 = df_1990_2019.rename(columns={"1990_x":"co2 1990", 
                                            "1990_y":"gdp 1990", 
                                            "2019_x":"co2 2019", 
                                            "2019_y":"gdp 2019"})
print(df_1990_2019)
print(df_1990_2019.describe())
pd.plotting.scatter_matrix(df_1990_2019, figsize=(12, 12), s=5, alpha=0.8)



# Clustering  co2 emissions--------------------------------------------------


# clustering of co2 emissions for 1990 and 2019
df_co2_1990_2019 = df_1990_2019[["co2 1990", "co2 2019"]].copy()
print(df_co2_1990_2019)

# normalise
df_co2_1990_2019, df_min, df_max = ct.scaler(df_co2_1990_2019)

# calculate and print silhouette scores
print("\nsilhouette scores of co2 emissions for 1990 & 2019")
print("n    score")
# Loop over number of clusters
for n_cluster in range(2, 10):
    labels, cen = fit_clusters(df_co2_1990_2019, n_cluster)
    silhouette_score = skmet.silhouette_score(df_co2_1990_2019, labels)
    print(n_cluster, silhouette_score)

# Fit clusters
labels, cen = fit_clusters(df_co2_1990_2019, 3)

# Add cluster labels to the data
df_1990_2019['Cluster'] = labels

# Plot clusters (normalized data)
plot_clusters(df_co2_1990_2019, labels, cen, "co2 1990", "co2 2019", 
              "3 Clusters of co2 emissions for 1990 & 2019 normalized data")

# And new the plot for the unnormalised data.

# Backscale cluster centers
cen_backscaled = ct.backscale(cen, df_min, df_max)

# Plot clusters (unnormalized data)
plot_clusters(df_1990_2019, labels, cen_backscaled, "co2 1990", "co2 2019", 
              "3 Clusters of co2 emissions for 1990 & 2019 unnormalized data")

