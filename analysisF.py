import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# LOAD DATA


def load_data():

    df = pd.read_csv("testdata.csv")

    df["site"] = df["site"].astype("category")

    return df

# DATASET SUMMARY
def dataset_summary(df):

    summary = {}

    summary["shape"] = df.shape
    summary["head"] = df.head(10)
    summary["dtypes"] = df.dtypes
    summary["missing"] = df.isna().sum()
    summary["describe"] = df.describe()
    summary["correlation"] = df.corr(numeric_only=True)

    return summary

# REVENUE ANALYSIS
def revenue_analysis(df):

    revenue_summary = df["revenue"].describe()

# Histogram
    fig_hist, ax = plt.subplots(figsize=(7,5))

    ax.hist(df["revenue"], bins=30)

    ax.set_xlabel("Revenue")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Revenue")

# Boxplot
    fig_box, ax2 = plt.subplots(figsize=(5,5))

    ax2.boxplot(df["revenue"])

    ax2.set_xlabel("Revenue")

    return {"summary": revenue_summary,"histogram": fig_hist,"boxplot": fig_box}

# TOP ANALYSIS
def top_analysis(df):

    top_summary = df["top"].describe()

    fig, ax = plt.subplots(figsize=(7,5))

    ax.hist(df["top"], bins=30)

    ax.set_xlabel("TOP")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of TOP")

    return {"summary": top_summary,"histogram": fig}



# CATEGORICAL ANALYSIS
def categorical_analysis(df):

    
    # Browser %
    

    browser_pct = (df["browser"].value_counts(normalize=True)*100)

    
    # Platform %
    

    platform_pct = (df["platform"].value_counts(normalize=True)*100)

    
    # Site %
    

    site_pct = (df["site"].value_counts(normalize=True)*100)

    
    # Browser Figure
    

    fig_browser, ax = plt.subplots(figsize=(7,5))

    ax.bar(browser_pct.index,browser_pct.values)

    ax.set_xlabel("Browser")
    ax.set_ylabel("%")

    ax.set_title("Distribution of Browser")

    
    # Platform Figure
    

    fig_platform, ax = plt.subplots(figsize=(7,5))

    ax.bar(platform_pct.index,platform_pct.values)

    ax.set_xlabel("Platform")
    ax.set_ylabel("%")

    ax.set_title("Distribution of Platform")

    
    # Site Figure
    

    fig_site, ax = plt.subplots(figsize=(7,5))

    ax.bar(site_pct.index,site_pct.values)

    ax.set_xlabel("Site")
    ax.set_ylabel("%")

    ax.set_title("Distribution of Site")

    return {"browser_pct":browser_pct,"platform_pct":platform_pct,"site_pct":site_pct,"browser_plot":fig_browser,"platform_plot":fig_platform,"site_plot":fig_site}


