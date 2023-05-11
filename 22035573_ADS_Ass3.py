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
    im2 = axes[1].matshow(corr2, cmap='viridis')
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


def analyze_clusters(df):
    """
    Analyze clusters based on the provided DataFrame.

    Args:
        df (DataFrame): Input DataFrame containing the data.

    Returns:
        None
    """
    # Print the dataframe containing values of 1990 and 2019
    print(df)

    # Print the representative countries from each cluster
    for cluster_id in range(3):
        cluster_countries = df[df["Cluster"] == cluster_id]
        representative_country = cluster_countries.sample(n=1)
        print(f"Cluster {cluster_id}: Rep Country - {representative_country['Country Name'].values[0]}")

    # Collect mean CO2 emissions for each cluster
    cluster_means = []
    for cluster_id in range(3):
        cluster_countries = df[df["Cluster"] == cluster_id]
        cluster_mean = cluster_countries[["co2 1990", "co2 2019"]].mean()
        cluster_means.append(list(cluster_mean))

    # Plot mean CO2 emissions for each cluster
    plt.figure(figsize=(10, 8))
    plt.bar(["1990", "2019"], cluster_means[2], 
            label="Cluster 2", alpha=0.5, color='b')
    plt.bar(["1990", "2019"], cluster_means[1], 
            label="Cluster 1", alpha=0.7, color='g')
    plt.bar(["1990", "2019"], cluster_means[0], 
            label="Cluster 0", alpha=0.9, color='y')

    plt.xlabel("Year")
    plt.ylabel("Mean CO2 Emissions")
    plt.title("Comparison of Mean CO2 Emissions of 1990 & 2019 for Different Clusters")
    plt.legend()
    plt.show()


def linear(x, a, b):
    """
    Linear function: f(x) = a + b*x
    
    Args:
        x (float or array-like): Input variable(s)
        a (float): Intercept parameter
        b (float): Slope parameter
        
    Returns:
        float or array-like: Value of the linear function at x
    """
    
    f = a + b*x
    
    return f


def fit_and_plot(df, x_label, y_label):
    """
    Perform curve fitting and 
    plot the best fitting function with confidence range.

    Args:
        df (DataFrame): Input DataFrame containing the data.
        x_label (str): Label of the x-axis column.
        y_label (str): Label of the y-axis column.

    Returns:
        None
    """
    
    # Define x values and observed data (CO2 emissions or GDP)
    x_values = df[x_label]
    y_values = df[y_label]

    # Perform curve fitting
    popt, pcov = opt.curve_fit(linear, x_values, y_values)

    # Generate predicted values using the fitted parameters
    df['fit'] = linear(x_values, *popt)

    # Calculate lower and upper confidence ranges using err_ranges
    lower, upper = err.err_ranges(x_values, linear, popt, 
                                  np.sqrt(np.diag(pcov)))

    # Plot the best fitting function and confidence range
    plt.figure(figsize=(12, 10))

    plot_clusters(df, labels, cen_backscaled, x_label, y_label, "fitting")

    plt.plot(x_values, df['fit'], "k--", label='Best Fit')

    plt.fill_between(x_values, lower, upper, alpha=0.8, 
                     label='Confidence Range', color='b')

    # add x, y labels and title
    plt.xlabel('Year 1990')
    plt.ylabel('Year 2019')
    plt.title(f'Fitting {x_label} vs {y_label}')
    
    # plot legend and show the plot
    plt.legend()
    plt.show()

