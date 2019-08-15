### Importing Libraries 
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from itertools import combinations

from scipy import stats
import scipy.cluster.hierarchy as shc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import mean_squared_error, confusion_matrix, auc, roc_curve
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif, \
    SelectPercentile, SelectFwe
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import power_transform, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif, \
    SelectPercentile, SelectFwe
from sklearn.feature_selection import chi2 as chii2

from xgboost import XGBClassifier

### Loading Libraries

# Path to the data sets
path = ""
study_a = pd.read_csv(path + "Study_A.csv")
study_b = pd.read_csv(path + "Study_B.csv")
study_c = pd.read_csv(path + "Study_C.csv")
study_d = pd.read_csv(path + "Study_D.csv")
study_e = pd.read_csv(path + "Study_E.csv")

# Creating a dataframe that consists of all studies
study = pd.concat([study_a, study_b, study_c, study_d], axis=0)

# An array of all columns names
columns = study.columns
# An array of all numerical columns names
numerical = study[['PatientID', 'Country', 'SiteID', 'RaterID',
                   'VisitDay', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'N1', 'N2', 'N3',
                   'N4', 'N5', 'N6', 'N7', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8',
                   'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'PANSS_Total']]

# LeadStatus Series
lead_status = study["LeadStatus"]
# TxGroup All and E study treatment dummies
treat_study = pd.get_dummies(study["TxGroup"])["Treatment"]
treat_e = pd.get_dummies(study_e["TxGroup"])["Treatment"]
# LeadStatus dummy
lead_dummy = pd.get_dummies(study["LeadStatus"])
lead_pass = lead_dummy["Passed"]
lead_failed = lead_dummy["Assign to CS"] + lead_dummy["Flagged"]
# Country dummy
dummy_country = pd.get_dummies(study["Country"])


# ADDS TOTAL POS/NEG, CHRONIC POS/NEG, BIPOLAR INDEX, GENERAL FEATURES
# RETURNS DB A-E
def add_general_features():
    studies = [study_a, study_b, study_c, study_d, study_e]

    for index, study_k in enumerate(studies):

        positive = study_k.iloc[:, 8:15]
        negative = study_k.iloc[:, 15:22]
        general = study_k.iloc[:, 22:38]

        total_positive = positive.sum(axis=1)
        total_negative = negative.sum(axis=1)
        total_general = general.sum(axis=1)

        bipolar_index = total_positive - total_negative

        chronic_positive = (np.sum(positive >= 4, axis=1) >= 3).astype(int)
        chronic_negative = (np.sum(negative >= 4, axis=1) >= 3).astype(int)

        scores = pd.concat(
            [total_positive, total_negative, bipolar_index, total_general, chronic_positive, chronic_negative], axis=1)

        data_up = pd.concat([studies[index], scores], axis=1).rename(
            columns={0: "Pos_Total", 1: "Neg_Total", 2: "Bipolar_index", 3: "General", 4: "Pos_Chronic",
                     5: "Neg_Chronic"})
        new_data = data_up
        studies[index] = new_data
    return studies


# STUDY DUMMIES A-D, E is out,
def data_num_vars():
    a, b, c, d, e = add_general_features()
    studies = pd.concat([a, b, c, d], axis=0)
    #STUDY dummies
    study_dummies = pd.get_dummies(studies["Study"])
    study_dummies.insert(0, 'E', 0)
    #study_dummies.insert(4, 'E', 0)
    studies = pd.concat([study_dummies, studies.iloc[:, 1:]], axis=1)

    #STUDY E dummies
    e_dummies = pd.get_dummies(e["Study"])
    e_dummies.insert(0, 'D', 0)
    e_dummies.insert(0, 'C', 0)
    e_dummies.insert(0, 'B', 0)
    e_dummies.insert(0, 'A', 0)
    #DELETE E in E
    e = pd.concat([e_dummies, e.iloc[:, 1:]], axis=1)
    #TREATMENT dummy
    studies["TxGroup"] = treat_study
    e["TxGroup"] = treat_e
    return studies, lead_pass, e


def data_question2():
    # x - All datasets without E, y - lead status for all datasets without E, e - dataset e
    x, y, e = data_num_vars()
    x = x.drop(["LeadStatus"], axis=1)
    data = pd.concat([x, e], axis=0)
    # STUDY A-D
    data = data[data["VisitDay"] == 0]
    return data, y

def q3():
    # x - All datasets without E, y - lead status for all datasets without E, e - dataset e
    x, y, e = data_num_vars()

    # COUNTRY DUMMY
    country_dummy = pd.get_dummies(e["Country"], drop_first=True)
    e = pd.concat([country_dummy, e], axis=1)

    # Drop Country
    e = e.drop(["A", "B", "C", "D", "Country"], axis=1)

    # Filter the 379 patients we have to predict
    sample_sub = pd.read_csv("sample_submission_PANSS (1).csv")
    patients_sample = sample_sub["PatientID"]
    e = e[e["PatientID"].isin(patients_sample)]

    # MEAN - adding Rater mean score
    rater_mean = e.groupby("RaterID")["PANSS_Total"].mean().round(1)
    e_join_rater_mean = e.join(rater_mean, on='RaterID', how='left', rsuffix="_Rater_Mean")
    e_join_rater_mean.rename(columns={'PANSS_Total_Rater_Mean': 'Rater_Mean'}, inplace=True)
    e = e_join_rater_mean

    # MEAN - adding Site mean score
    site_mean = e.groupby(["SiteID"])["PANSS_Total"].mean().round(1)
    e_join_site_mean = e.join(site_mean, on='SiteID', how='left', rsuffix="_Site_Mean")
    e_join_site_mean.rename(columns={'PANSS_Total_Site_Mean': 'Site_Mean'}, inplace=True)
    e = e_join_site_mean

    # STD - adding Site std score
    site_std = e.groupby("SiteID")["PANSS_Total"].std().round(1)
    e_join_site_std = e.join(site_std, on='SiteID', how='left', rsuffix="_Site_Std")
    e_join_site_std.rename(columns={'PANSS_Total_Site_Std': 'Site_Std'}, inplace=True)
    e = e_join_site_std

    # ORDER COLUMNS
    use_columns = ['PatientID', 'PANSS_Total', 'VisitDay', 'TxGroup', 'Rater_Mean', 'Site_Mean', 'Site_Std', 'Pos_Total',
                   'Neg_Total', 'Bipolar_index', 'General', 'Pos_Chronic',
                   'Neg_Chronic', 'USA', 'UK', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'N1', 'N2',
                   'N3', 'N4', 'N5', 'N6', 'N7', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7',
                   'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16']

    e = e[use_columns]
    return e


def q3a():
    # x - All datasets without E, y - lead status for all datasets without E, e - dataset e
    x, y, e = data_num_vars()
    data = pd.concat([x, e], axis=0)

    # Creating Time_pct_change column
    data["new_col"] = data.groupby("PatientID")["VisitDay"].rank()
    # PCT CHANGE
    data["Total_pct_change"] = data.groupby("PatientID")["PANSS_Total"].pct_change().fillna(0)
    # PCT CHANGE VISIT
    data["VisitDay"] = data["VisitDay"].replace(0, 1)
    data["Time_pct_change"] = data.groupby("PatientID")["VisitDay"].pct_change(periods=-1).fillna(0)

    # Add mean Total
    patient_mean = data.groupby(["PatientID"])["PANSS_Total"].mean().round(3)
    data_join_patient_mean = data.join(patient_mean, on='PatientID', how='left', rsuffix="_Patient_Mean")
    data_join_patient_mean.rename(columns={'PANSS_Total_Patient_Mean': 'Patient_Mean'}, inplace=True)
    data = data_join_patient_mean

    ##RATERID
    # DATA
    rater_mean = data.groupby("RaterID")["PANSS_Total"].mean().round(3)
    data_join_rater_mean = data.join(rater_mean, on='RaterID', how='left', rsuffix="_Rater_Mean")
    data_join_rater_mean.rename(columns={'PANSS_Total_Rater_Mean': 'Rater_Mean'}, inplace=True)
    data = data_join_rater_mean

    ##SITE
    # DATA
    site_mean = data.groupby(["SiteID"])["PANSS_Total"].mean().round(3)
    data_join_site_mean = data.join(site_mean, on='SiteID', how='left', rsuffix="_Site_Mean")
    data_join_site_mean.rename(columns={'PANSS_Total_Site_Mean': 'Site_Mean'}, inplace=True)
    data = data_join_site_mean
    # COUNTRIES

    ##COUNTRY MEAN
    # DATA
    country_mean = data.groupby(["Country"])["PANSS_Total"].mean().round(3)
    data_join_country_mean = data.join(country_mean, on='Country', how='left', rsuffix="_Country_Mean")
    data_join_country_mean.rename(columns={'PANSS_Total_Country_Mean': 'Country_Mean'}, inplace=True)
    data = data_join_country_mean

    # E
    country_europe = ['Ukraine', 'Spain', 'Romania', 'UK',
                      'Czech Republic', 'Poland', 'Portugal', 'Bulgaria', 'Germany',
                      'Greece', 'Slovakia', 'France', 'Hungary', 'Belgium', 'Austria', 'Sweden']
    country_americas = ['Brazil', 'Mexico', 'Canada', 'Argentina']
    country_asia = ["ERROR", 'India', 'Korea', 'China', 'Taiwan', 'Japan', 'Australia']
    for i in country_asia:
        data = data.replace({'Country': i}, "Asia")
    for i in country_europe:
        data = data.replace({'Country': i}, "Europe")
    for i in country_americas:
        data = data.replace({'Country': i}, "Americas")

    # COUNTRY DUMMIES BY CONTINENT
    country_dummies = pd.get_dummies(data.Country).drop("Americas", axis=1)
    data = pd.concat([country_dummies, data], axis=1)
    return data


def q3b():
    data = q3a()

    e = data[data["E"] == 1]

    y = lead_status

    x = data[data["E"] != 1]
    ex = data[data["E"] == 1]

    # Out flagged study D
    data = pd.concat([x, y], axis=1)
    dataD = data[data["D"] == 1]
    dataD = dataD[dataD["LeadStatus"] != 'Flagged']
    data = data[data["D"] != 1]
    data = pd.concat([data, dataD], axis=0)

    # OUT C europe
    data = data[data["A"] != 1]
    data = data.iloc[:, :-1]
    data = pd.concat([data, ex], axis=0)

    # ADDING FOLLOWING SCORE
    data["Next_Total"] = data.groupby("PatientID").PANSS_Total.shift(-1)
    data = data.dropna()
    y = data["Next_Total"]
    data = data.drop(["Next_Total"], axis=1)

    # E
    sample = pd.read_csv("sample_submission_PANSS (1).csv")
    patients = sample["PatientID"]
    e = e[e["PatientID"].isin(patients)]
    e = e.groupby("PatientID").last().reset_index()
    columns = data.columns
    data = data[columns]
    e = e[columns]
    data = data.drop(["Country", "PatientID", "SiteID", "RaterID"], axis=1)
    e = e.drop(["Country", "PatientID", "SiteID", "RaterID"], axis=1)
    return data, y, e


def q3c():
    data = q3a()
    e = data[data["E"] == 1]
    data["Last_score"] = data.groupby("PatientID")["PANSS_Total"].transform('last')
    y = data["Last_score"]
    data = data.drop(["Last_score"], axis=1)
    data = data.drop(["Country", "PatientID", "SiteID", "RaterID"], axis=1)
    e = e.drop(["Country", "PatientID", "SiteID", "RaterID"], axis=1)
    return data, y, e


def q4():
    # x - All datasets without E, y - lead status for all datasets without E, e - dataset e
    x, y, e = data_num_vars()

    # Country Mean Total
    country_mean = x.groupby(["Country"])["PANSS_Total"].mean().round(3)
    data_join_country_mean = x.join(country_mean, on='Country', how='left', rsuffix="_Country_Mean")
    data_join_country_mean.rename(columns={'PANSS_Total_Country_Mean': 'Country_Mean'}, inplace=True)
    x = data_join_country_mean
    country_mean = e.groupby(["Country"])["PANSS_Total"].mean().round(3)
    data_join_country_mean = e.join(country_mean, on='Country', how='left', rsuffix="_Country_Mean")
    data_join_country_mean.rename(columns={'PANSS_Total_Country_Mean': 'Country_Mean'}, inplace=True)
    e = data_join_country_mean

    # Country Mean Bipolar index
    country_mean = x.groupby(["Country"])["Bipolar_index"].mean().round(3)
    data_join_country_mean = x.join(country_mean, on='Country', how='left', rsuffix="_Country_Mean_BI")
    data_join_country_mean.rename(columns={'PANSS_Total_Country_Mean_BI': 'Country_Mean_BI'}, inplace=True)
    x = data_join_country_mean
    country_mean = e.groupby(["Country"])["Bipolar_index"].mean().round(3)
    data_join_country_mean = e.join(country_mean, on='Country', how='left', rsuffix="_Country_Mean_BI")
    data_join_country_mean.rename(columns={'PANSS_Total_Country_Mean_BI': 'Country_Mean_BI'}, inplace=True)
    e = data_join_country_mean

    country_europe = ['Ukraine', 'Spain', 'Romania',
                      'Czech Republic', 'Poland', 'Portugal', 'Bulgaria', 'Germany',
                      'Greece', 'Slovakia', 'France', 'Hungary', 'Belgium', 'Austria', 'Sweden']
    country_americas = ['Brazil', 'Mexico', 'Canada', 'Argentina']
    country_asia = ["ERROR", 'India', 'Korea', 'China', 'Taiwan', 'Japan', 'Australia']
    for i in country_asia:
        x = x.replace({'Country': i}, "Asia")
    for i in country_europe:
        x = x.replace({'Country': i}, "Europe")
    for i in country_americas:
        x = x.replace({'Country': i}, "Americas")

    # COUNTRY DUMMIES BY CONTINENT AND USA & RUSIA
    country_dummies = pd.get_dummies(x.Country)  # .drop("Europe", axis=1)
    x = pd.concat([country_dummies, x], axis=1)
    e_country_dummies = pd.get_dummies(e["Country"]).drop("UK", axis=1)
    e_country_dummies.insert(0, 'Asia', 0)
    e_country_dummies.insert(0, 'Americas', 0)
    e_country_dummies.insert(0, 'Europe', 0)
    e = pd.concat([e_country_dummies, e], axis=1)

    # ALL RATER COUNTRY
    x["Rater_USA"] = pd.DataFrame(x["USA"] == 1).astype(int)
    x["Rater_Russia"] = pd.DataFrame(x["Russia"] == 1).astype(int)
    x["Rater_Americas"] = pd.DataFrame(x["Americas"] == 1).astype(int)
    x["Rater_Asia"] = pd.DataFrame(x["Asia"] == 1).astype(int)

    # E RATER COUNTRY
    e["Rater_USA"] = pd.DataFrame(e["USA"] == 1).astype(int)
    e["Rater_Russia"] = pd.DataFrame(e["Russia"] == 1).astype(int)
    e.insert(0, 'Rater_Asia', 0)
    e.insert(0, 'Rater_Americas', 0)

    # PCT CHANGE
    x["Total_pct_change"] = x.groupby("PatientID")["PANSS_Total"].pct_change().fillna(0)
    e["Total_pct_change"] = e.groupby("PatientID")["PANSS_Total"].pct_change().fillna(0)

    # SITEMEAN
    # DATA
    site_mean = x.groupby(["SiteID"])["PANSS_Total"].mean().round(3)
    data_join_site_mean = x.join(site_mean, on='SiteID', how='left', rsuffix="_Site_Mean")
    data_join_site_mean.rename(columns={'PANSS_Total_Site_Mean': 'Site_Mean'}, inplace=True)
    x = data_join_site_mean
    # E
    site_mean = e.groupby(["SiteID"])["PANSS_Total"].mean().round(3)
    e_join_site_mean = e.join(site_mean, on='SiteID', how='left', rsuffix="_Site_Mean")
    e_join_site_mean.rename(columns={'PANSS_Total_Site_Mean': 'Site_Mean'}, inplace=True)
    e = e_join_site_mean

    ##RATERIDMEAN
    # DATA
    rater_mean = x.groupby("RaterID")["PANSS_Total"].mean().round(3)
    data_join_rater_mean = x.join(rater_mean, on='RaterID', how='left', rsuffix="_Rater_Mean")
    data_join_rater_mean.rename(columns={'PANSS_Total_Rater_Mean': 'Rater_Mean'}, inplace=True)
    x = data_join_rater_mean
    # E
    rater_mean = e.groupby("RaterID")["PANSS_Total"].mean().round(3)
    e_join_rater_mean = e.join(rater_mean, on='RaterID', how='left', rsuffix="_Rater_Mean")
    e_join_rater_mean.rename(columns={'PANSS_Total_Rater_Mean': 'Rater_Mean'}, inplace=True)
    e = e_join_rater_mean

    # INTERACTION VARIABLES X
    x["ranked_visit"] = x.groupby("PatientID")["VisitDay"].rank()
    x["Bi-pos-sq"] = x["Pos_Total"] ** 2 - x["Neg_Total"]
    x["Bi-neg-sq"] = x["Pos_Total"] - x["Neg_Total"] ** 2
    x["Bi-sq"] = x["Bipolar_index"] ** 2
    x["G12-G8"] = x["G12"] - x["G8"]
    x["G14-G8"] = x["G14"] - x["G8"]
    x["P2-P5"] = x["P2"] - x["P5"]
    x["P3-G15"] = x["P3"] - x["G15"]

    x["P3-P1-SQ"] = x["P3"] ** (2) - x["P1"] ** 2
    x["P1-N1"] = x["P1"] - x["N1"]
    x["P1-P1-SQ"] = x["P1"] ** (2) - x["N1"] ** 2
    x["G2-G4"] = x["G2"] - x["G4"]
    x["N6-N3"] = x["N6"] - x["N3"]
    x["N5-N1"] = x["N5"] - x["N1"]
    x["N5-G3"] = x["N5"] - x["G3"]
    x["G2-G4-SQ"] = (x["G2"] - x["G4"]) ^ (2)
    x["G2SQ-G4SQ"] = x["G2"] ^ 2 - x["G4"] ^ 2
    # INTERACTION VARIABLES
    e["ranked_visit"] = e.groupby("PatientID")["VisitDay"].rank()
    e["Bi-pos-sq"] = e["Pos_Total"] ** 2 - e["Neg_Total"]
    e["Bi-neg-sq"] = e["Pos_Total"] - e["Neg_Total"] ** 2
    e["Bi-sq"] = e["Bipolar_index"] ** (2)
    e["G12-G8"] = e["G12"] - e["G8"]
    e["P2-P5"] = e["P2"] - e["P5"]
    e["G14-G8"] = e["G14"] - e["G8"]
    e["P3-G15"] = e["P3"] - e["G15"]

    e["P3-P1-SQ"] = e["P3"] ** (2) - e["P1"] ** (2)
    e["P1-N1"] = e["P1"] - e["N1"]
    e["P1-P1-SQ"] = e["P1"] ** (2) - e["N1"] ** (2)
    e["G2-G4"] = e["G2"] - e["G4"]
    e["N6-N3"] = e["N6"] - e["N3"]
    e["N5-N1"] = e["N5"] - e["N1"]
    e["N5-G3"] = e["N5"] - e["G3"]
    e["G2-G4-SQ"] = (e["G2"] - e["G4"]) ^ (2)
    e["G2SQ-G4SQ"] = e["G2"] @ - e["G4"] ^ 2
    # VALUES
    x = x.drop(["AssessmentiD", "Country", "RaterID", "PatientID", "SiteID"], axis=1)
    e = e.drop(["AssessmentiD", "Country", "RaterID", "PatientID", "SiteID"], axis=1)
    x = x[sorted(x.columns)]
    e = e[sorted(e.columns)]
    return x, y, e


def q4a():
    # x - All datasets without E, y - lead status for all datasets without E, e - dataset e
    x, y, e = q4()
    x = x.drop("LeadStatus", axis=1)

    # TAKING OUT STUDY D FLAGGED
    y = lead_status
    data = pd.concat([x, y], axis=1)
    dataD = data.iloc[-2948:, :]
    dataD = dataD[dataD["LeadStatus"] != 'Flagged']
    data = data.iloc[:-2948, :]
    data = pd.concat([data, dataD], axis=0)

    # Taking out Asia PASSED
    datac = data[data["C"] == 1]
    datanoc = data[data["C"] != 1]
    dataceu = datac[datac["Asia"] == 1]
    datacnoeu = datac[datac["Asia"] != 1]
    dataceu = dataceu[dataceu["LeadStatus"] != 'Passed']
    dataceu = pd.concat([dataceu, datacnoeu], axis=0)
    data = pd.concat([datanoc, dataceu], axis=0)

    # OUT C europe
    datanoc = data[data["C"] != 1]
    datac = data[data["C"] == 1]
    datacnoeu = datac[datac["Europe"] != 1]
    data = pd.concat([datanoc, datacnoeu], axis=0)

    # Putting it back
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # ADDING BACK Y, TAKING OUT FLAGGED A
    lead_dummy = pd.get_dummies(y)
    y = lead_dummy["Passed"]
    return x, y, e


def pairwise_plot(data, columns):
    pair_var = [[[i, j] for j in columns] for i in columns]
    for i in pair_var:
        for j in i:
            sns.lmplot(j[0], j[1],
                       data=data,
                       fit_reg=False,
                       hue="cluster",
                       scatter_kws={"marker": "D",
                                    "s": 100})
            plt.title('Clusters {} vs {}'.format(j[0], j[1]))
            plt.xlabel(j[0])
            plt.ylabel(j[1])


# Apply RFE on nums best parameters, based on model
def rfe(x, y, model, nums):
    from sklearn.feature_selection import RFE
    features = nums
    for i in range(1, features):
        rfe = RFE(model, i)
        xrfe = rfe.fit_transform(x, y)
        x_train, x_test, y_train, y_test = train_test_split(xrfe, y, test_size=0.20, random_state=0)
        model.fit(x_train, y_train)
        print(i, model.score(x_test, y_test))
        ranking = rfe.ranking_
        print(ranking)
        tag = rfe.support_
    return tag, ranking


def chi2(x, y, k):
    bestfeatures = SelectKBest(score_func=chii2, k=k)
    fit = bestfeatures.fit(x, y)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x.columns)

    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    chi2 = featureScores.nlargest(k, 'Score')
    chi = featureScores.nlargest(k, 'Score')["Specs"]
    chi2 = chi2.reset_index(drop=True)
    return chi2, chi


def fclass(x, y, num):
    bestfeatures = SelectKBest(score_func=f_classif, k=num)
    fit = bestfeatures.fit(x, y)
    scores = pd.DataFrame(fit.scores_)
    pvalues = pd.DataFrame(fit.pvalues_)
    columns = pd.DataFrame(x.columns)

    # concat two dataframes for better visualization
    featureScores = pd.concat([columns, scores, pvalues], axis=1)
    featureScores.columns = ['Specs', 'Score', 'Pvalues']
    f_class = featureScores.nlargest(num, 'Score')
    f = featureScores.nlargest(num, 'Score')["Specs"]
    f_class = f_class.reset_index(drop=True)
    return f_class, f


def mutual(x, y, num):
    bestfeatures = SelectKBest(score_func=mutual_info_classif, k=10)
    fit = bestfeatures.fit(x, y)
    scores = pd.DataFrame(fit.scores_)
    columns = pd.DataFrame(x.columns)
    featureScores = pd.concat([columns, scores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    mutual = featureScores.nlargest(num, 'Score')
    mut = featureScores.nlargest(num, 'Score')["Specs"]
    mutual = mutual.reset_index(drop=True)
    return mutual, mut


def logreg(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    logreg = LogisticRegression().fit(x_train, y_train)

    y_pred = logreg.predict(x_test)
    score = logreg.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print("score: ", score)
    print("MSE: ", mse)
    return


def tree(x, y, e, depth):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=0)
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    score = model.score(x_test, y_test)
    print("CV score: ", np.mean(cross_val_score(model, x_test, y_test, cv=5)))
    print("Trees score: ", score)
    print("MSE: ", mean_squared_error(y_test, predict))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predict)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    probabilities = model.predict_proba(e)
    print("ROC_AUC: ", roc_auc)
    print(confusion_matrix(y_test, predict))
    probabilities = model.predict_proba(e)
    return probabilities[:, 1]


def forest(x, y, e):
    model = RandomForestClassifier(bootstrap=True, criterion='entropy',
                                   max_depth=22, max_features="sqrt", n_estimators=1, min_samples_split=0.1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    score = model.score(x_test, y_test)
    print("CV score: ", np.mean(cross_val_score(model, x_test, y_test, cv=5)))
    print("Forest score: ", score)
    print("MSE: ", mean_squared_error(y_test, predict))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predict)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    probabilities = model.predict_proba(e)
    print("ROC_AUC: ", roc_auc)
    print("CONFUSION MATRIX: \n", confusion_matrix(y_test, predict))
    predicted = model.predict(e)
    pr = model.predict_proba(x_test)
    return probabilities[:, 0], predicted, predict, pr[:, 1]


def gbc(x, y, e):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20)
    model = GradientBoostingClassifier().fit(x_train, y_train)
    predict = model.predict(x_test)
    score = model.score(x_test, y_test)
    print("CV score: ", np.mean(cross_val_score(model, x_test, y_test, cv=5)))
    print("GBC Classifier: ", score)
    print("MSE: ", mean_squared_error(y_test, predict))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predict)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    probabilities = model.predict_proba(e)
    print("ROC_AUC: ", roc_auc)
    print("CONFUSION MATRIX: \n", confusion_matrix(y_test, predict))
    predicted = model.predict(e)
    print(pd.Series(predicted).value_counts())
    print(pd.Series(predict).value_counts())
    return probabilities[:, 0], predicted, predict


def xgb(x, y, e):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y)
    model = XGBClassifier()

    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_weighted', 'roc_auc']
    scores = cross_validate(model, x_train, y_train, scoring=scoring, cv=10)
    sorted(scores.keys())
    forest_accuracy = scores['test_accuracy'].mean()
    forest_precision = scores['test_precision_macro'].mean()
    forest_recall = scores['test_recall_macro'].mean()
    forest_f1 = scores['test_f1_weighted'].mean()
    forest_roc = scores['test_roc_auc'].mean()
    print("forest_accuracy: ", forest_accuracy)
    print("forest_precision: ", forest_precision)
    print("forest_recall: ", forest_recall)
    print("forest_f1: ", forest_f1)
    print("forest_roc: ", forest_roc)

    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    score = model.score(x_test, y_test)
    print("XGB Classifier: %.2f%%" % (score * 100.0))
    print("MSE: ", mean_squared_error(y_test, predict))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predict)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    probabilities = model.predict_proba(e)
    print("ROC_AUC: ", roc_auc)
    print("CONFUSION MATRIX: \n", confusion_matrix(y_test, predict))

    predicted = model.predict(e)
    print(pd.Series(predicted).value_counts())
    print(pd.Series(predict).value_counts())
    return probabilities[:, 0], predicted, predict


def predict(predictions):
    ids = study_e["AssessmentiD"]
    proba = pd.Series(predictions)
    data = pd.concat([ids, proba], axis=1)
    data = data.rename(columns={0: "LeadStatus"})
    return data


def predict_regression(predictions):
    data = pd.read_csv("sample_submission_PANSS (1).csv")
    ids = data.iloc[:, 0]
    proba = pd.Series(predictions)
    data = pd.concat([ids, proba], axis=1)
    data = data.rename(columns={0: "PANSS_Total"})
    return data


def onehot(x):
    one = preprocessing.OneHotEncoder()
    one.fit(x)
    one_hot = one.transform(x).toarray()
    return one_hot


# Standardising the DataFrame
def NormalDataFrame(df):
    return (df - df.mean()) / df.std()


# Normalising the DataFrame
def MinMaxDataFrame(df):
    return (df - df.min()) / (df.max() - df.min())


## Project Part B

# Print clusters after use of GMM using ClusterN as number of clusters, DataSet - to cluster,
# PrintDataSet - to produce plots. Outputs the cluster labels for DataSet.
def printClusterGMM(DataSet, ClusterN, PrintDataSet):
    labels_pairs = list(combinations(["PANSS_Total", "COMP", "GENR"], 2))

    cluster = GaussianMixture(n_components=ClusterN, random_state=0)
    labels = cluster.fit_predict(DataSet)

    plt.figure(figsize=(40, 10))
    gs = gridspec.GridSpec(1, len(labels_pairs))

    for i in range(len(labels_pairs)):
        name1, name2 = labels_pairs[i]
        ax = plt.subplot(gs[0, i])
        plt.title("Clustering {} using {}, {}".format("GMM", name1, name2))
        plt.scatter(PrintDataSet[name1], PrintDataSet[name2], c=labels, cmap='rainbow')
    print("FINISHED")
    return labels


# Print clusters after use of K-Means using ClusterN as number of clusters, DataSet - to cluster,
# PrintDataSet - to produce plots. Outputs the cluster labels for DataSet.
def printClusterKM(DataSet, ClusterN, PrintDataSet):
    labels_pairs = list(combinations(["PANSS_Total", "COMP", "GENR"], 2))

    cluster = KMeans(n_clusters=ClusterN, random_state=0)
    cluster.fit(DataSet)

    plt.figure(figsize=(40, 10))
    gs = gridspec.GridSpec(1, len(labels_pairs))

    for clustern in range(ClusterN):
        print("Cluster {} has {} observations".format(clustern + 1, sum(cluster.labels_ == clustern)))

    for i in range(len(labels_pairs)):
        name1, name2 = labels_pairs[i]
        ax = plt.subplot(gs[0, i])
        plt.title("Clustering {} using {}, {}".format("KMeans", name1, name2))
        plt.scatter(PrintDataSet[name1], PrintDataSet[name2], c=cluster.labels_, cmap='rainbow')
    print("FINISHED")
    return cluster.labels_


# Print clusters after use of Hierrachy using ClusterN as number of clusters and linkage_type,
# DataSet - to cluster, PrintDataSet - to produce plots. Outputs the cluster labels for DataSet.
def printClusterHierarchy(linkage_type, DataSet, ClusterN, PrintDataSet):
    labels_pairs = list(combinations(["PANSS_Total", "COMP", "GENR"], 2))

    cluster = AgglomerativeClustering(n_clusters=ClusterN, affinity='euclidean', linkage=linkage_type)
    cluster.fit_predict(DataSet)

    for clustern in range(ClusterN):
        print("Cluster {} has {} observations".format(clustern + 1, sum(cluster.labels_ == clustern)))

    plt.figure(figsize=(40, 10))
    gs = gridspec.GridSpec(1, len(labels_pairs))

    for i in range(len(labels_pairs)):
        name1, name2 = labels_pairs[i]
        ax = plt.subplot(gs[0, i])
        plt.title("Clustering {} using {}, {}".format(linkage_type, name1, name2))
        plt.scatter(PrintDataSet[name1], PrintDataSet[name2], c=cluster.labels_, cmap='rainbow')
    print("FINISHED")
    return cluster.labels_


# Prints histograms for each cluster using the column names we provide.
# Gives no output.
def quality_of_cluster(data, clusters, ClusterN, names_list=["PANSS_Total", "COMP", "GENR", "POSS", "NEGG"]):
    colors = ['b', 'g', 'y', 'r', 'o', 'p'] * 10
    bins = np.arange(0, 8, 1 / ClusterN)
    colors = colors[0:ClusterN]

    plt.figure(figsize=(6, 3))
    for name in names_list:
        mean = [data[clusters == i][name].mean() for i in range(1, ClusterN + 1)]
        var = [data[clusters == i][name].var() for i in range(1, ClusterN + 1)]
        print("For {} we have:".format(name))
        for i in range(ClusterN):
            print("Cluster {} - Mean: {} Var: {}".format(i + 1, mean[i], var[i]))
            if data[clusters == i][name].unique().size < 9:
                plt.hist(data[clusters == i][name] + i / ClusterN, bins, color=colors[i],
                         label="Cluster {}".format(i + 1), alpha=0.5)
            else:
                plt.hist(data[clusters == i][name], color=colors[i], label="Cluster {}".format(i + 1), alpha=0.5)
        plt.legend(loc='upper right')
        plt.title("For Predictor {}".format(name))
        plt.show()


## Project Part C

# Performs K-Means clustering for out patients
def findClosest(NameA, DataSet, k=5):

    # Finds the euclidean distance between Patients A and B on day 0
    def distanceB(Name1, Name2):
        distance = 0
        for i in range(8, 39):
            distance += (DataSet.iloc[:, i][(DataSet["PatientID"] == Name1)].mean() - DataSet.iloc[:, i][
                (DataSet["PatientID"] == Name2)].mean()) ** 2
        return distance

    DataSet_Zero = DataSet[DataSet["VisitDay"] == 0].reset_index()
    DataSet_DistA = pd.DataFrame({"PatientID": DataSet_Zero["PatientID"].unique()})
    DataSet_DistA["Distance"] = [distanceB(NameA, NameB) for NameB in DataSet_DistA["PatientID"]]

    k = min(k, DataSet_DistA.shape[0])

    BestNames = DataSet_Zero["PatientID"].tail(k)
    return BestNames


# Provides predictions of the next visit for all the patients in the DataSet
# Uses KNN and Linear Regression for predicting
# Outputs DataFrame - PatientID and Predictor
def SimpleUniqueLinear(DataSet):

    def printProgress(i):
        print("-" * 20)
        print("Progress: %8.2f%%" % (i / len(PatientIDs) * 100))


    # Mapping to allow weighted combination of Reg and KMeans prediction
    def MapingUniversal(k):
        return np.exp((1 - k) / 2)

    PatientIDs = DataSet["PatientID"].unique()
    DataSet_R = pd.DataFrame({"PatientID": PatientIDs, 'PANSS_Total': [0] * len(PatientIDs)})

    for i in range(len(PatientIDs)):

        mask = DataSet["PatientID"] == PatientIDs[i]
        amount_days = DataSet["VisitDay"][mask].shape[0]
        koef = MapingUniversal(amount_days)

        low_boandary, high_boandary = 119, 127
        LinReg_Pred = 0

        # Linear Regression PREDICTION
        if amount_days >= 2:

            reg = LinearRegression().fit(np.array(DataSet.loc[mask, "VisitDay"]).reshape(-1, 1),
                                         DataSet.loc[mask, "PANSS_Total"])
            amount_days = DataSet["VisitDay"][mask].shape[0]

            pred_day = 123
            # If more than 2 observations than we calculate a special date
            if amount_days > 2:
                pred_day = DataSet.loc[mask, "VisitDay"].max() + (
                            DataSet.loc[mask, "VisitDay"][1:amount_days - 1] - DataSet.loc[mask, "VisitDay"][
                                                                               0:amount_days]).mean()
                low_boandary, high_boandary = pred_day - 4, pred_day + 4

            LinReg_Pred = reg.predict(np.array([pred_day]).reshape(1, -1))

        # KNN PREDICTION
        useful_patients = [PatientID for PatientID in PatientIDs if (
                    (DataSet["PatientID"] == PatientID) & (DataSet["VisitDay"] > low_boandary) & (
                        DataSet["VisitDay"] < high_boandary)).sum() > 0]
        useful_mask = (DataSet["VisitDay"] == 0) & (DataSet["PatientID"].isin(useful_patients))

        while useful_mask.sum() < 1:
            koef *= 2 / 3
            low_boandary, high_boandary = low_boandary - 4, high_boandary + 4

            useful_patients = [PatientID for PatientID in PatientIDs if (
                        (DataSet["PatientID"] == PatientID) & (DataSet["VisitDay"] > low_boandary) & (
                            DataSet["VisitDay"] < high_boandary)).sum() > 0]
            useful_mask = (DataSet["VisitDay"] == 0) & (DataSet["PatientID"].isin(useful_patients))

        BestFits = findClosest(PatientIDs[i], DataSet[useful_mask])
        KNN_Pred = DataSet[DataSet["PatientID"].isin(BestFits)]["PANSS_Total"].mean()

        # Averaging the predictions using the inverse weight of amount of days

        DataSet_R.ix[i, "PANSS_Total"] = float(koef * KNN_Pred + (1 - koef) * LinReg_Pred)
        printProgress(i + 1) if not (i + 1) % 5 else None

    print("Finished!")
    return DataSet_R


# Produces RSS score to compare with previous models that were saved
# Version - version of the last saved model
# Returns int value - RSS score
def CompRSS_Previous(New_Sub, versions):
    RSS_1 = [0] * versions
    for version in range(versions):
        Prev_Sub = pd.read_csv(path + "prediction_{}.csv".format(version + 1))

        for rown in range(New_Sub.shape[0]):
            RSS_1[version] += (New_Sub["PANSS_Total"][rown] - Prev_Sub["PANSS_Total"][rown]) ** 2

        RSS_1[version] /= New_Sub.shape[0]

    return RSS_1

### Project part C - unfinished model

# Adds the visit time for the DataSet
def addVisitTime(DataSet):
    VisitCol = [-1] * DataSet.shape[0]

    PatientID = -1
    VisitN = 0
    for rown in range(DataSet.shape[0]):
        if PatientID != DataSet.iloc[rown]["PatientID"]:
            VisitN = 1
            PatientID = DataSet.iloc[rown]["PatientID"]
        VisitCol[rown] = VisitN
        VisitN += 1

    NewDataSet = DataSet.copy()
    NewDataSet["VisitTime"] = VisitCol

    return NewDataSet


# Clean repeated measurements for a patient in a day
# by substituting them with the mean of all the measurements in the day
def CleanRepeates(DataSet):
    DataSet_Clean = pd.DataFrame(columns=DataSet.columns)

    counter = 1
    saved_row = DataSet.iloc[0]
    start = 1
    for rown in range(1, DataSet.shape[0]):
        if not start and saved_row["VisitDay"] == DataSet.iloc[rown]["VisitDay"] and saved_row["PatientID"] == \
                DataSet.iloc[rown]["PatientID"]:
            counter += 1
            saved_row[8:39] += DataSet.iloc[rown][8:39]
            start = 0

        else:
            saved_row[8:39] /= counter
            DataSet_Clean = DataSet_Clean.append(saved_row)
            saved_row = DataSet.iloc[rown]

            counter = 1

        print("Done %8.2f%%" % ((rown + 1) / DataSet.shape[0] * 100)) if not (rown + 1) % 200 else None

    return DataSet_Clean


# Adding the smart change to the DataSet
def AddSmartChange(DataSet):
    # Requires VisitTime
    NewDataSet = DataSet.copy()
    NewDataSet["SmartChange"] = [0.] * DataSet.shape[0]

    place = 0
    value = 0
    start = 1
    for rown in range(0, DataSet.shape[0]):
        if DataSet.iloc[rown]["VisitDay"] == 0 and not start:
            if DataSet.iloc[rown - 1]["VisitDay"] != 0:
                NewDataSet["SmartChange"][place] = value / DataSet.iloc[rown - 1]["VisitTime"]
            place = rown
            value = 0
        else:
            start = 0
            value += (DataSet.iloc[rown]["PANSS_Total"] - DataSet.iloc[rown - 1]["PANSS_Total"]) * DataSet.iloc[rown][
                "VisitTime"]

    NewDataSet["SmartChange"][place] = value / DataSet.iloc[DataSet.shape[0] - 1]["VisitTime"]
    return NewDataSet

