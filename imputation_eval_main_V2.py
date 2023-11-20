# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:34:54 2023

@author: lucas
"""

import numpy as np
import pandas as pd
import os
#from sklearn.datasets import fetch_california_housing
import miceforest as mf
import matplotlib.pyplot as plt
#import torch
import time
#import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
#import tqdm
from scipy import stats
from scipy.linalg import LinAlgError
#from scipy.optimize import basinhopping
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from classification_Prototype import find_mode_of_kde
from kde_helper import GaussianKdeWithEpsilon
from copy import deepcopy 
import pickle 
import cloudpickle 
from sklearn.decomposition import PCA


def delete_randomly_data(df, delete_percent, random_state=None):
    num_values_to_delete = int(df.size * delete_percent)
    np.random.seed(random_state)
    indices = np.random.choice(df.size, num_values_to_delete, replace=False)
    
    df_copy = df.copy(deep=True).values.flatten()
    if df_copy.dtype.kind == 'f':
        df_copy[indices] = np.nan
    else:
        df_copy = df_copy.astype(float)
        df_copy[indices] = np.nan
    
    return pd.DataFrame(df_copy.reshape(df.shape), columns=df.columns)

def plot_missing_data(df, filename,save=False):
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]

    if missing_data.empty:
        print("Keine fehlenden Daten vorhanden.")
        return

    missing_data.sort_values(ascending=False, inplace=True)
    plt.bar(missing_data.index, missing_data.values)

    plt.title('Anzahl der fehlenden Daten pro Spalte')
    plt.xlabel('Spalte')
    plt.ylabel('Anzahl fehlender Daten')
    plt.xticks(rotation=45)
    if save == True:
        filename = filename + "-delete-analysis.png"
        plt.savefig(filename)
    plt.show()
    
def distribution_mice(num_iter,kernel,column_pos,instance):
  # get the imputed values from the multiple imputed datasets
  imput_values = []
  for i in range(0,num_iter):
    imput_df = kernel.complete_data(dataset=i)
    try:
        column = imput_df.iloc[:,column_pos]
    except:
        column = imput_df[:,column_pos]
    row = column[instance]
    imput_values.append(row)
  return imput_values

def pdf_variance(points):
    # Calculate the PDF using a histogram with 10 bins
    hist, bin_edges = np.histogram(points, bins=100, density=True)
    
    # Calculate the mean of the PDF
    mean = np.mean(hist)
    
    # Calculate the variance of the PDF
    variance = np.sum((hist - mean)**2 * np.diff(bin_edges))
    
    return variance

def fill_kde_dict(kde_dict,size):
    # Erzeuge ein neues OrderedDict, um die Reihenfolge beizubehalten
    sorted_dict = OrderedDict()

    # Iteriere über die Zahlen von 1 bis 49
    for num in range(0, size-1):
        # Überprüfe, ob die Zahl bereits im kde_dict vorhanden ist
        if num not in kde_dict:
            # Füge den Key-Wert-Paar dem sorted_dict hinzu
            sorted_dict[num] = np.nan

    # Füge die vorhandenen Key-Wert-Paare aus kde_dict in sorted_dict ein
    for key, value in kde_dict.items():
        sorted_dict[key] = value

    return sorted_dict
    
def KDE_evaluation(X, X_complete, imputer, num_samples_mice, num_samples_kde, filename):
    np.set_printoptions(suppress=True)
    instance_kde_dim = X.shape[1]
    # calculate percentage of deleted values across all attributes
    perc_dele = X.isna().sum().sum()
    total_dele = X.size
    percentage = (perc_dele / total_dele) * 100
    mse_per_dim = np.zeros(instance_kde_dim)
    deviation_per_dim = np.zeros(instance_kde_dim)
    count_per_dim = np.zeros(instance_kde_dim)
    variance = 0
    count_pdfs = 0
    shift_list = []
    df_mode = X.copy()
    df = X.copy()
    columns = df.columns
    kde_models_dict = {}
    for index, x in df.iterrows():
        instance_kde = []
        missing_dims = np.where(np.isnan(x))[0]  # Get the indices of missing dimensions
        if len(missing_dims) > 0:
            for i in range(len(x)):
                if i in missing_dims:
                    # extract the estimated points form the imputation for every dimension
                    # print(X_complete.iloc[index,i])
                    mice_imp_points = np.array(distribution_mice(num_samples_mice, imputer, i, index))
                    mice_array = mice_imp_points.reshape((num_samples_mice,))
                    # print(mice_array.shape, type(mice_array))
                    instance_kde.append(mice_array)
                    # evaluate the variance of the pdf
                    variance += pdf_variance(mice_imp_points)
                    count_pdfs += 1
#-------------------------------------------------------------------------------
            # define multinomial kernel density estimation
            instance_kde = np.array(instance_kde).T
            instance_kde = instance_kde.T
            # print(instance_kde)
            # print(instance_kde.shape)
#---------------------------------------------------------------    
            # create KDE
            '''
            # Annahme: Ihr DataFrame heißt 'data'
            threshold = 0.001  # Schwellenwert für die Standardabweichung
            
            # Berechnung der Standardabweichung für jede Dimension
            std_dev = instance_kde.std()
            
            # Identifikation konstanter Dimensionen
            constant_dims = std_dev[std_dev < threshold].index
            
            # Entfernung konstanter Dimensionen
            instance_kde = instance_kde.drop(columns=constant_dims)
            '''
            #kde = stats.gaussian_kde(dataset=instance_kde, bw_method=0.2)
            kde = GaussianKdeWithEpsilon(instance_kde)
            #kde = stats.gaussian_kde(dataset=instance_kde)
            # save kde in dict as well as index of corresponding instance
            kde_models_dict[index] = (kde,0)
#------------------------------------------------------------------
            # create samples - correlation is taken into account between samples
            modal_values = find_mode_of_kde(kde)
            # print("modal values: ",modal_values)
            # add modal values to x (instance) containing missing values
            # print(type(modal_values))
            modal_count = 0
            for i in missing_dims:
                df_mode.iloc[index, i] = modal_values[modal_count]
                modal_count += 1
            # print(df_mode.iloc[0])
#---------------------------------------------------------------------------   
            # evaluation    
            # find the true values for the missing dimensions
            real_values = X_complete.iloc[index,missing_dims].values
            for i in range(len(missing_dims)):  
                # Calculate shift percentage
                shift_percentage = ((real_values[i]  - modal_values[i]+1e-4) / (modal_values[i]+1e-4)) * 100
                # print(shift_percentage)
                # Add deviation percentage to result list
                shift_list.append(shift_percentage)
            # print(shift_list)
            for i in range(len(missing_dims)):
                # Calculate the MSE for each dimension
                mse = np.sqrt(mean_squared_error([modal_values[i]], [real_values[i]]))
                mse_per_dim[missing_dims[i]] += mse
                if abs(real_values[i]) > 0.0:
                    deviation = np.abs((modal_values[i] - real_values[i]) / (real_values[i])) * 100
                else:
                    deviation = 0.0
                deviation_per_dim[missing_dims[i]] += deviation
                count_per_dim[missing_dims[i]] += 1
                print("mse: ",mse, "deviation: ",deviation)
#------------------------------------------------------------------------
        else:
            # x containts no missing value - just add x once
            kde_models_dict[index] = np.nan
    
    # safe kdes as pkl file
    kde_models_dict = fill_kde_dict(kde_models_dict,len(X))
    kde_models_dict = dict(sorted(kde_models_dict.items(), key=lambda x: x[0]))
    # Calculate average MSE per dimension
    # print("before: ", deviation_per_dim, "mse_per_dim: ", mse_per_dim)
    mse_per_dim = np.divide(mse_per_dim, count_per_dim)
    deviation_per_dim = np.divide(deviation_per_dim, count_per_dim)
    print(deviation_per_dim)
    print(mse_per_dim)
    # Plot erstellen
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(deviation_per_dim)), deviation_per_dim)
    
    # Beschriftungen hinzufügen
    for i, deviation in enumerate(deviation_per_dim):
        rmse = np.sqrt(mse_per_dim[i])
        ax.text(i, deviation + 1, f'RMSE: {rmse:.2f}', ha='center')
    
    # Achsenbeschriftungen und Titel setzen
    plt.xlabel('Dimension')
    plt.ylabel('Percentage of deviation')
    plt.title('Average RMSE/Deviation per Dimension (Modal-Value)')
    
    # Achsengrenzen festlegen
    #hier die nan-Werte durch 0 ersetzen 
    deviation_per_dim[np.where(np.isnan(deviation_per_dim))]=0
    print("Deviation per dim", deviation_per_dim)
    plt.ylim(0, max(deviation_per_dim) + 10)
    
    # Dateiname für das Speichern des Plots
    filename = filename + "-average_rmse_deviation.png"
    # Plot anzeigen
    plt.show()
    # Plot speichern
    plt.savefig(filename)
    
    # average variance across all pdfs
    if count_pdfs > 0:
        avg_variance = variance / count_pdfs
    else:
        avg_variance = 0.0
    
    # Num of bins for value separation
    num_bins = 40
    
    # point value to bin
    bins = np.linspace(-100, 100, num_bins+1)
    
    # count values per bin
    hist, _ = np.histogram(shift_list, bins=bins)
    
    # Plot
    plt.hist(shift_list, bins=bins, edgecolor='black')
    plt.xlabel('Deviation from Mode (%)')
    plt.ylabel('Number of PDFs')
    plt.title('Histogramm')
    text1 = f"The average variance of the PDF is: {round(avg_variance,3)}"
    plt.text(0.97, 0.8, text1, ha='right', va='top', transform=plt.gca().transAxes,fontsize=8)
    # text2 = f"The mse is: {round(mse_total,3)}"
    # plt.text(0.97, 0.7, text2, ha='right', va='top', transform=plt.gca().transAxes,fontsize=8)
    text3 = f"Percentage of deleted data: {round(percentage, 0)}%"
    plt.text(0.97, 0.9, text3, ha='right', va='top', transform=plt.gca().transAxes, fontsize=8)
    
    # Set y-axis limit
    plt.ylim(0, max(hist) + 10)
    
    filename = filename + "-imputation-analysis.png"
    # plt.savefig(filename)
    plt.show()
    
    return df_mode, kde_models_dict

def feature_importance(X,y):
    try:
        # Initialisation of Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
        plt.barh(importance['Feature'], importance['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.show()
    except:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
        plt.barh(importance['Feature'], importance['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.show()
        
        
        
        

if __name__ == "__main__":
    
    #dataset = "credit"
    #dataset = "adult"
    #dataset = "heart"
    dataset = "housing"
    #dataset = None
    
    if dataset == "diabetes":
        diabetes = pd.read_csv("mice_evaluation/diabetes.csv")
        diabetes = diabetes.sample(frac=1, random_state=42)
        diabetes.dropna(inplace=True)
        # Delete Data (over all Attributes)
        X = diabetes.drop(columns='Outcome')
        X_complete = diabetes.drop(columns='Outcome')
        X_shape = X.shape[1]
        y = diabetes["Outcome"]
        filename = ""

        
    elif dataset == "wine":    
        wine_quality = pd.read_csv("mice_evaluation/Wine Quality.csv")
        # Delete Data (over all Attributes)
        X = wine_quality.drop(columns='quality')
        X_complete = wine_quality.drop(columns='quality')
        X_shape = X.shape[1]
        y = wine_quality["quality"]
        # where files should be safed
        filename = "C:/...."
        
    elif dataset == "credit":
        credit = pd.read_csv("credit/data/credit_data_transformed.csv")
        print(credit.head())
        print(credit.columns)
        credit = credit.sample(frac=1, random_state=42)
        credit.dropna(inplace=True)
        # Delete Data (over all Attributes)
        X = credit.drop(columns=['Approved']) #, 'Married', 'Gender', 'DriversLicense'
        X_complete = credit.drop(columns=['Approved'])
        X_shape = X.shape[1]
        y = credit["Approved"]
        filename = ""
        
    elif dataset == "adult":
        adult = pd.read_csv("adult/data/adult.csv", na_values=["?"])
        adult = adult.sample(frac=1, random_state=42)
        adult.dropna(inplace=True)
        # Delete Data
        X = adult.drop(columns=['income'])
        X_complete = adult.drop(columns=['income'])
        y = adult['income']
        y.reset_index(inplace=True, drop=True)
        categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

        # Wähle nur die kategorialen Spalten aus X aus
        X_categorical = X[categorical_columns]
        
        # Erstelle einen OrdinalEncoder und wende ihn auf die kategorialen Spalten an
        encoder = OrdinalEncoder()
        X_encoded_categorical = encoder.fit_transform(X_categorical)
        
        # Ersetze die kategorialen Spalten in X durch die codierten Werte
        X_encoded = X.copy()
        X_encoded[categorical_columns] = X_encoded_categorical
        X_encoded.reset_index(inplace=True, drop=True)
        X = X_encoded
        X_complete = X_encoded
        filename=""
    elif dataset == "heart":
        heart = pd.read_csv("heart/data/heart.csv")
        heart = heart.sample(frac=1, random_state=42)
        heart.dropna(inplace=True)
        X = heart.drop(columns=['target'])
        X_complete = heart.drop(columns=['target'])
        y = heart['target']
        y.reset_index(inplace=True, drop=True)
        filename = ""
    elif dataset == "housing":
        heart = pd.read_csv("housing/data/housing.csv")
        heart = heart.sample(frac=1, random_state=42)
        heart.dropna(inplace=True)
        X = heart.drop(columns=['Price'])
        X_complete = heart.drop(columns=['Price'])
        y = heart['Price']
        y.reset_index(inplace=True, drop=True)
        filename = ""

    PLATZHALTER = 'iris'
    MISSING = 30
    # Hier Pfad zum Datensatz angeben:
    # dataset = pd.read_csv("C:/Users/paulm/OneDrive/Desktop/Arbeit/Verteilungen einfügen/Datensätze/künstlich fehlend/crashes/pop_failures.dat", sep="\s+")
    if dataset == None:
        #with open(f"{PLATZHALTER}/data/{PLATZHALTER}_data_test.npy", 'rb') as f:
            #X = np.load(f)
            #y = np.load(f)
        iris = np.load('iris/data/iris.npy')
        X = iris[:, :-1]
        y = iris[:, -1]
    
    X = pd.DataFrame(X)
    print(X.head())
    
    # Für Australia Dataset
    if dataset == "credit":
        #X['Income'] = np.sqrt(X['Income'])
        corr = X.corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm')
        plt.show()
        
        threshold = 0.4  # Beispiel-Schwellenwert
        correlated_pairs = {}
        
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > threshold:
                    attribute1 = corr.columns[i]
                    attribute2 = corr.columns[j]
                    correlation_value = corr.iloc[i, j]
                    correlated_pairs[(attribute1, attribute2)] = correlation_value
        
        print("Stark korrelierte Attribute:")
        for pair, correlation_value in correlated_pairs.items():
            print(f"{pair[0]} und {pair[1]}: {correlation_value:.2f}")
            
        plt.figure(figsize=(12, 6))  # Legen Sie die Größe des Plots fest
        sns.boxplot(data=X)  # Erstellen Sie den Boxplot
        
        # Optional: Beschriften Sie die Achsen
        plt.xlabel("Attribute")  # Beschriften Sie die x-Achse
        plt.ylabel("Werte")  # Beschriften Sie die y-Achse
        plt.title("Boxplot zur Identifizierung von Ausreißern")  # Setzen Sie den Titel des Plots
        
        # Zeigen Sie den Plot an
        plt.show()
        #X.columns = np.arange(X.shape[1])
    
    
    X_complete = deepcopy(X)
    
    filename=""
    
    
     
#---------------------------------------------------------------------------------------------------
    # Feature imporance
    feature_importance(X, y)    
    
    if dataset == "credit":
        numeric_columns = ['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'Income']
        categorical_columns = ['Gender', 'Married', 'BankCustomer', 'Industry', 'Ethnicity', 'PriorDefault', 'Employed', 'DriversLicense', 'Citizen', 'ZipCode']
        
        numeric_features = X[numeric_columns]
        categorical_features = X.drop(columns=numeric_columns)
        
        #scaler = MinMaxScaler()
        scaler = StandardScaler()
    
        numeric_features_scaled = scaler.fit_transform(numeric_features)
        numeric_features_scaled_df = pd.DataFrame(numeric_features_scaled, columns=numeric_columns)
        
        X = pd.concat([numeric_features_scaled_df, categorical_features], axis=1)
        X_complete = deepcopy(X)
    
        # scaling
        # X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        # X_complete = pd.DataFrame(scaler.fit_transform(X_complete), columns=X_complete.columns)
        #X = pd.DataFrame(preprocessor.fit_transform(X), columns=X.columns)
        #X_complete = pd.DataFrame(preprocessor.fit_transform(X_complete), columns=X_complete.columns)
        print(X.shape)
        #X = pd.DataFrame(preprocessor.fit_transform(X), columns=X.columns)
        #X_complete = pd.DataFrame(preprocessor.fit_transform(X_complete), columns=X_complete.columns)
    
        # delete values 
        # Hier Namen der Spalten aus denen gelöscht werden soll eingeben:
        #bei Bands: 16-34 sind numerisch, wähle 16,19,22,25 zunächst einmal aus, optimale Spalten zu bestimmen   
        #echoc: 0,1,3,4,5,6,7
        #breast: beliebig 
        #old setting for breast dataset: cols_to_delete = [0,3,6,9,12], now only 0,3,6,9, need to find optimal ones here
        #cols_to_delete = [0,1,10,11]
        cols_to_delete = ["Gender", "Age", "CreditScore", "DriversLicense"]
        #cols_to_delete = list(range(X_complete.shape[1]))
    elif dataset == "adult":
        cols_to_delete = ["age", "capital-gain", "capital-loss", "hours-per-week"]
    elif dataset == "heart":
        cols_to_delete = ['age', 'trestbps', 'chol']
    elif dataset == "housing":
        cols_to_delete = ['Rooms', 'Distance', 'Propertycount']
    else:
        cols_to_delete = [0, 1, 3]
    
    
    X[cols_to_delete] = delete_randomly_data(X[cols_to_delete], MISSING*0.01, 31)
    # evaluate data delete
    plot_missing_data(X, filename)
    
#-------------------------------------------------------------------------------------------------------
    # create kernel on missing data
    d = 50
    # number of instances
    size = X.shape[0]
    print(f"Imputing data... (size: {size})")
    start_time = time.time()
    imputer = None
    if MISSING != 0:
        imputer = mf.ImputationKernel(
          np.array(X), # give Data with missing values
          datasets=d,
          save_all_iterations=True,
          random_state=1
        )
        imputer.mice(4)
    end_time = time.time()
    imputation_time = end_time - start_time
    print("Imputation completed in {:.2f} seconds.".format(imputation_time))
# # #---------------------------------------------------------------------------------------------------------------
    print("Sampling and Modal Value calculation...")
    
    start_time = time.time()
    ## create samples and impute the modal value of the pdf
    # df_sample, df_mode, avg_rmse, avg_variance = pdfc.sample_and_mode_mice(X, X_complete, "mice", imputer, 100, 100,filename)
    df_mode, kde_models_dict = KDE_evaluation(X[:size], X_complete[:size], imputer, d, d,filename)
    end_time = time.time()
    imputation_time = end_time - start_time
    print("Sampling and Modal Value calculation finished in {:.2f} seconds.".format(imputation_time))
    
    try:
        KDE1 = kde_models_dict[1][0]
        kernel_min = np.min(KDE1.dataset)
        kernel_max = np.max(KDE1.dataset)
        x1 = np.linspace(kernel_min, kernel_max, 1000)
        y1 = KDE1.pdf(x1)
        plt.plot(x1, y1)
    except:
        pass
    
    if dataset == None:
        #X = pd.concat([X, pd.DataFrame(y)], axis=1)
        #X=X.reindex(columns=list(range(X.columns.size)))
        X[X.columns.size] = y
        print(X)
        #X.join(y)
    elif dataset == "credit":
        X["approved"] = y
    elif dataset == "adult":
        X["income"] = y
    elif dataset == "heart":
        X["target"] = y
    elif dataset == "housing":
        X["Price"] = y
# -----------------------------------------------------------------------------------------------------
    #save results - kde and dataframe with imputed data
    
    if dataset == None:
        dataset = PLATZHALTER
    try:
        with open(f"{dataset}/data/{dataset}_saved_pdfs_{MISSING}.pkl", 'wb') as outp:
            pickle.dump(kde_models_dict, outp, pickle.HIGHEST_PROTOCOL)
    except:
        with open(f"{dataset}/data/{dataset}_saved_pdfs_{MISSING}.cp.pkl", 'wb') as outp:
            cloudpickle.dump(kde_models_dict,outp)
        os.remove(f"{dataset}/data/{dataset}_saved_pdfs_{MISSING}.pkl")

    try:
        with open(f"{dataset}/data/{dataset}_data_{MISSING}.pkl", 'wb') as outp:
            pickle.dump(X, outp, pickle.HIGHEST_PROTOCOL)
    except:
        with open(f"{dataset}/data/{dataset}_data_{MISSING}.cp.pkl", 'wb') as outp:
            cloudpickle.dump(X, outp)
    


