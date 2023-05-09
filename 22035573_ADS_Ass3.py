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
