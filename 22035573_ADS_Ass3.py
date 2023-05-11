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
        df_years, df_countries (pd.DataFrame): 
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

    return df_years, df_countries


def plot_corr_heatmap_scatter_matrix(df1, df2, title1, size=6):
    """
    Plots correlation heatmaps for a dataframe and a scatter matrix plot.

    Args:
        df1 (pandas DataFrame): Input dataframe for the heatmap
        df2 (pandas DataFrame): Input dataframe for the scatter matrix
        title1 (str): Title for the heatmap
        size (int): The vertical and horizontal size of the plot (in inches)
    """

    # Compute the correlation matrix of df1
    corr = df1.corr()

    # Plot correlation heatmap
    plt.figure(figsize=(size, size))
    plt.matshow(corr, cmap='coolwarm')

    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    # Set the title
    plt.title('Correlation heatmap of CO2 Emissions')

    # Add a colorbar to the plot and show the plot
    plt.colorbar()
    plt.show()

    # Plot scatter matrix for df2
    pd.plotting.scatter_matrix(df2, figsize=(12, 12), s=5, alpha=0.8)

    # show scatter matrix
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

    # Create a K-means clustering model with the specified number of clusters
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)

    # Fit the data to the K-means model
    kmeans.fit(df)

    # Get the cluster labels assigned to each data point
    labels = kmeans.labels_

    # Get the coordinates of the cluster centroids
    cen = kmeans.cluster_centers_

    # Return the cluster labels and centroids
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

    # Plots the scatter cluster points
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


def analyze_clusters(df):
    """
    Analyze clusters based on the provided DataFrame.

    Args:
        df (DataFrame): Input DataFrame containing the data.

    Returns:
        None
    """
    # Print the df containing values of 1990, 2019 and extra clusters column
    print(df)

    # Print the representative countries from each cluster
    print("\n Print the representative countries from each cluster : ")
    for cluster_id in range(3):
        cluster_countries = df[df["Cluster"] == cluster_id]
        representative_country = cluster_countries.sample(n=1)
        print(
            f"Cluster {cluster_id}: Rep Country - {representative_country['Country Name'].values[0]}")

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
            label="Cluster 1", alpha=0.9, color='r')
    plt.bar(["1990", "2019"], cluster_means[0],
            label="Cluster 0", alpha=0.5, color='g')

    # add x, y labels
    plt.xlabel("Year", fontsize=15)
    plt.ylabel("Mean CO2 Emissions", fontsize=15)

    # add title, legend and show the plot
    plt.title("Comparison of Mean CO2 Emissions of 1990 & 2019 for all 3 Clusters",
              fontsize=15)
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
    And predict.
    Args:
        df (DataFrame): Input DataFrame containing the data.
        x_label (str): Label of the x-axis column.
        y_label (str): Label of the y-axis column.

    Returns:
        None
    """

    # Define x values and observed data (CO2 emissions)
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

    # Plot the scatter points of the data
    plot_clusters(df, labels, cen_backscaled, x_label, y_label, "fitting")

    # Plot the best fit line
    plt.plot(x_values, df['fit'], "k--", label='Best Fit')

    # Plot the confidence range
    plt.fill_between(x_values, lower, upper, alpha=0.8,
                     label='Confidence Range', color='b')

    # add x, y labels and title
    plt.xlabel('Year 1990', fontsize=15)
    plt.ylabel('Year 2019', fontsize=15)
    plt.title('Best Fit of CO2 emissions in 1990 vs 2019 with confidence range and Predictions',
              fontsize=17)

    # Generate x values for future years
    x_future = np.array([2029, 2039, 2049])  # Example future years
    x_future = x_future-2000

    # Predict corresponding y values for future years using fitted parameters
    y_pred = linear(x_future, *popt)

    # Calculate lower and upper confidence ranges for the predicted values
    lower_pred, upper_pred = err.err_ranges(x_future, linear, popt,
                                            np.sqrt(np.diag(pcov)))

    # Plot the predicted values and confidence range
    plt.plot(x_future, y_pred, 'r-', label='Predicted (2029, 2039, 2049)')
    plt.fill_between(x_future, lower_pred, upper_pred, alpha=0.3,
                     label='Prediction Uncertainty Range')

    # plot legend and show the plot
    plt.legend()
    plt.show()


# Main Program


# Reading Files ----------------------------------------------------------
# read the data for "CO2 emissions (metric tons per capita)"
df_co2, df_co2_years = read_df("co2 emissions.csv")


# Summary Statistics-------------------------------------------------------


# summary statistics for "CO2 emissions(metric tons per capita)" of whole world
print("\nCO2 emissions summary statistics for whole world:")
print(df_co2.describe())

# Select specific years for analysis
df_co2_y = df_co2[[1990, 1995, 2000, 2005, 2010, 2015, 2019]]

# Plot bar chart comparing CO2 emissions of selected countries over the years
df_co2.iloc[46:52].plot.bar(x="Country Name",
                            y=[1990, 2000, 2019],
                            figsize=(8, 8), xlabel='Countries', rot=30,
                            ylabel='co2 emissions',
                            title='Comaprision of co2 emissions of few countries over the years')

# drop rows with NaN in the given year
df_co2 = df_co2.dropna(subset=[1990, 2019])

# Create a new DataFrame for the selected years
df_1990_2019 = df_co2[["Country Name", 1990, 2019]].copy()

# Drop rows with NaN values in the selected years
df_1990_2019 = df_1990_2019.dropna()

# rename columns
df_1990_2019 = df_1990_2019.rename(
    columns={1990: "co2 1990", 2019: "co2 2019"})

# Print the new dataframe
print(df_1990_2019)

# Print summary statistics for the new DataFrame
print("\nCO2 emissions summary statistics for whole world in 1990 and 2019:")
print(df_1990_2019.describe())

# Plot correlation heatmaps and scatter matrix for CO2
plot_corr_heatmap_scatter_matrix(df_co2_y, df_1990_2019, "CO2 Emissions")


# Clustering  CO2 Emissions--------------------------------------------------


# Select CO2 emissions data for 1990 and 2019 , print it
df_co2_1990_2019 = df_1990_2019[["co2 1990", "co2 2019"]].copy()
print(df_co2_1990_2019)

# Normalise the data
df_co2_1990_2019, df_min, df_max = ct.scaler(df_co2_1990_2019)


# calculate and print silhouette scores
print("\nsilhouette scores of co2 emissions for 1990 & 2019")
print("n    score")
# Loop over number of clusters
for n_cluster in range(2, 10):
    labels, cen = fit_clusters(df_co2_1990_2019, n_cluster)
    silhouette_score = skmet.silhouette_score(df_co2_1990_2019, labels)
    print(n_cluster, silhouette_score)


# Fit clusters with the optimal number of clusters (3 in this case)
labels, cen = fit_clusters(df_co2_1990_2019, 3)

# Add cluster labels to the data
df_1990_2019['Cluster'] = labels

# Plot clusters (normalized data)
plot_clusters(df_co2_1990_2019, labels, cen, "co2 1990", "co2 2019",
              "3 Clusters of co2 emissions for 1990 vs 2019 normalized data")

# And new the plot for the unnormalised data.

# Backscale cluster centers
cen_backscaled = ct.backscale(cen, df_min, df_max)

# Plot clusters (unnormalized data)
plot_clusters(df_1990_2019, labels, cen_backscaled, "co2 1990", "co2 2019",
              "3 Clusters of co2 emissions for 1990 vs 2019 unnormalized data")


# Clustering  CO2 Emissions Trends------------------------------------------


# Analyze clusters based on CO2 emissions trends
analyze_clusters(df_1990_2019)


# Fitting CO2 Emissions -----------------------------------------------------


# Perform curve fitting and plot the best fitting function
# with confidence range and predictions
fit_and_plot(df_1990_2019, 'co2 1990', 'co2 2019')

