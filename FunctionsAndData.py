import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d


#----------------Extracting all main data for import ------------------
tr_data = []  # Empty list to hold TR data
vl_data = []  # Empty list to hold VL data
memorization_score = []  # Empty list to hold memorization scores
data_path = 'data' 
for root, dirs, files in os.walk(data_path):
    if root.endswith('_TR.eval') or root.endswith('_VL.eval'): # Check if we are in a TR.eval or VL.eval folder
        for file in files:
            if file.startswith('eval_loss_meta'): # Check if the file is 'eval_loss_meta'
                file_path = os.path.join(root, file)
                # Read the file into a DataFrame
                try: 
                    df = pd.read_csv(file_path)
                # Append to the appropriate list
                    if root.endswith('_TR.eval'):
                        tr_data.append(df)
                    elif root.endswith('_VL.eval'):
                          vl_data.append(df)
                except:
                    print(f'Error reading {file_path}')
                
tr_df = pd.concat(tr_data, ignore_index=True)
vl_df = pd.concat(vl_data, ignore_index=True)
tr_avg_f1 = tr_df.groupby(['idx', 'moltype'])['f1'].mean().reset_index() #Keeping moltype column
tr_avg_f1.rename(columns={'f1': 'tr_avg_f1'}, inplace=True)
vl_avg_f1 = vl_df.groupby(['idx'])['f1'].mean().reset_index()
vl_avg_f1.rename(columns={'f1': 'vl_avg_f1'}, inplace=True)
memscore_df = pd.merge(tr_avg_f1, vl_avg_f1, on='idx')
memscore_df['Memorization_Score'] = (memscore_df['tr_avg_f1'] - memscore_df['vl_avg_f1'])/((memscore_df['tr_avg_f1'] + memscore_df['vl_avg_f1'])) # Subtract the VL F1 score from the TR F1 score for each idx




#--------------------Extracting Data for Upsample data for importing into the main file---------------------
tr_data_upscore = []
vl_data_upscore = []  
memorization_score_upscore = [] 
for root, dirs, files in os.walk(data_path):
    # Check if we are in a TR.eval or VL.eval folder within a sub-folder ending with '_upsample_rand50'
    if (root.endswith('_TR.eval') or root.endswith('_VL.eval')) and '_upsample_rand50' in root:
        for file in files:
            if file.startswith('eval_loss_meta'):  # Check if the file is 'eval_loss_meta'
                file_path = os.path.join(root, file)
                # Read the file into a DataFrame
                try:
                    df = pd.read_csv(file_path)
                    # Append to the appropriate list based on folder name
                    if root.endswith('_TR.eval'):
                        tr_data.append(df)
                    elif root.endswith('_VL.eval'):
                        vl_data.append(df)
                except Exception as e:
                    print(f'Error reading {file_path}: {e}')

tr_df_upscore = pd.concat(tr_data, ignore_index=True)
vl_df_upscore = pd.concat(vl_data, ignore_index=True)
tr_avg_f1_upscore = tr_df_upscore.groupby(['idx', 'moltype'])['f1'].mean().reset_index() #Keeping moltype column
tr_avg_f1_upscore.rename(columns={'f1': 'tr_avg_f1'}, inplace=True)
vl_avg_f1_upscore = vl_df_upscore.groupby(['idx'])['f1'].mean().reset_index()
vl_avg_f1_upscore.rename(columns={'f1': 'vl_avg_f1'}, inplace=True)
memscore_df_upsample = pd.merge(tr_avg_f1_upscore, vl_avg_f1_upscore, on='idx')# Combining the TR and VL dataframes into one
memscore_df_upsample['Memorization_Score'] = (memscore_df_upsample['tr_avg_f1'] - memscore_df_upsample['vl_avg_f1'])/((memscore_df_upsample['tr_avg_f1'] + memscore_df_upsample['vl_avg_f1']))



#----------------- Importing Pickele data for the main file---------------------
with open('/Users/ilanastern/Documents/GitHub/aiConfidential/nr80-vs-nr80.rnaforester.alnIdentity_pairwise.pkl', 'rb') as file:
    data = pickle.load(file)
pickle_df = pd.DataFrame(data)  # Renamed file to 'pickle_df' and converting to pandas
pickle_df.index = pickle_df.index.map(lambda x: str(x).split('_', 1)[0]) # Rename idx values to stop before the first underscore so idx values match up to memorization score data
pickle_df.columns = pickle_df.columns.map(lambda x: str(x).split('_', 1)[0])
pickle = pickle_df
print(pickle)


#------------------- Generating a Kernal Density Dataframe from the pickle data---------------------
# input the pickle package into the function to pull out the kde scores based on the pickle data
def kde_scores(dataframe):
    def kde(row):
        row = 1 - row  # Invert the row values
        sigma = np.std(row)
        mu = np.mean(row)
        f = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((row - mu) ** 2) / sigma ** 2)
        return f

    kde_scores = {}
    for idx, row in dataframe.iterrows(): # Iterate through each row in the DataFrame and compute the mean KDE score
        row_values = row.values
        kde_values = kde(row_values)  # Vectorized KDE for the entire row
        kde_scores[idx] = np.mean(kde_values)  # Compute the mean KDE score for the row

    kde_scores_df = pd.DataFrame(list(kde_scores.items()), columns=['idx', 'similarity_score']) # Convert the results to a DataFrame
    return kde_scores_df

#-------------------Plotter Function for any two dataframes---------------------
def exp_decay(x, a, b, c):# Exponential decay function
        return a * np.exp(-b * x) + c

def plotter(memscoredf, simscoredf, simscore_name = 'Data Frame Name'): #df1 is memscore, df2 is the simmilarity score
    memscoredf['idx'] = memscoredf['idx'].astype(str)# Ensuring 'idx' columns are the same data type
    simscoredf['idx'] = simscoredf['idx'].astype(str)
    merged_df = pd.merge(memscoredf, simscoredf, on='idx', how='inner')# Merging DataFrames on 'idx'
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['similarity_score', 'Memorization_Score'])

    # Fitting data
    x = merged_df['similarity_score']
    y = merged_df['Memorization_Score']
    initial_guess = [1, 1, 1]
    params, covariance = curve_fit(exp_decay, x, y, p0=initial_guess)
    x_fit = np.linspace(min(x), max(x), 500)
    y_fit = exp_decay(x_fit, *params)

    # Unique moltypes and color mapping
    moltypes = merged_df['moltype'].unique()
    color_map = {moltype: color for moltype, color in zip(moltypes, plt.cm.tab10.colors[:len(moltypes)])}

    # Plot 1: Memorization Score vs Distribution Score (KDE by Moltype)
    plt.figure(figsize=(10, 6))
    for moltype in moltypes:
        subset = merged_df[merged_df['moltype'] == moltype]
        plt.scatter(subset['similarity_score'], subset['Memorization_Score'],
                    color=color_map[moltype], label=f'Moltype {moltype}')
    plt.plot(x_fit, y_fit, color='red', label=f'Exp. Decay fit: {params[0]:.2f} * exp(-{params[1]:.2f} * x) + {params[2]:.2f}')
    plt.xlabel('Distribution Score')
    plt.ylabel('Memorization Score')
    plt.yscale('log')
    plt.title(f'Memorization Score vs Distribution Score ({simscore_name} by Moltype)')
    plt.legend()
    plt.show()
    print(f'The exponential decay equation is: y = {params[0]:.2f} * e^(-{params[1]:.2f} * x) + {params[2]:.2f}')

    # Plot 2: Best Fit Line with Error Bars
    plt.figure(figsize=(10, 6))
    perr = np.sqrt(np.diag(covariance))  # Standard deviation of parameters
    y_err = exp_decay(x_fit, *(params + perr)) - exp_decay(x_fit, *params)
    plt.plot(x_fit, y_fit, color='red', label=f'Best Fit: y = {params[0]:.2f} * exp(-{params[1]:.2f} * x) + {params[2]:.2f}')
    plt.fill_between(x_fit, y_fit - y_err, y_fit + y_err, color='gray', alpha=0.3, label='Fit Uncertainty')
    plt.xlabel('Similarity Score')
    plt.ylabel('Memorization Score')
    plt.yscale('log')
    plt.title(f'Memorization Score vs Similarity Score ({simscore_name} with Uncertainty)')
    plt.legend()
    plt.show()

    # Plot 3: Bubble Chart to Show Density of Points
    plt.figure(figsize=(10, 6))
    density, x_edges, y_edges = np.histogram2d(x, y, bins=50)
    x_center = (x_edges[:-1] + x_edges[1:]) / 2
    y_center = (y_edges[:-1] + y_edges[1:]) / 2
    x_mesh, y_mesh = np.meshgrid(x_center, y_center)
    plt.scatter(x_mesh.ravel(), y_mesh.ravel(), s=density.ravel() * 10, alpha=0.6, label='Density of Points')
    plt.plot(x_fit, y_fit, color='red', label=f'Fit: y = {params[0]:.2f} * exp(-{params[1]:.2f} * x) + {params[2]:.2f}')
    plt.xlabel('Similarity Score')
    plt.ylabel('Memorization Score')
    plt.title(f'Memorization Score vs Similarity Score ({simscore_name} Bubble Chart)')
    plt.legend()
    plt.show()



#-------------------Exponential Decay Function for the pickle data--------------------
def exponential_sim_scores(pickle_df):
    # Constants for the exponential decay function
    A = 10  # Maximum points for the highest score (close to 1)
    B = np.log(A) / (1 - 0.85)  # Solve for B so that a score of 0.85 maps to 1 point

    def map_similarity_to_points(score):
        if pd.isna(score) or score >= 1 or score < 0.0:  # Exclude NaN and out-of-bound values
            return 0
        return A * np.exp(-B * (1 - score))  # Apply exponential decay function

    similarity_scores = {} # Calculate similarity scores for each row
    for idx, row in pickle_df.iterrows():  # Iterate over each row by index
        points = row.apply(map_similarity_to_points).sum()  # Sum points in the row, ignoring NaN
        similarity_scores[idx] = points  # Store the total points for each idx
    similarity_scores_df = pd.DataFrame(list(similarity_scores.items()), columns=['idx', 'similarity_score'])# Convert the results to a DataFrame
    return similarity_scores_df


#-------------------- Quanta model of Simmilarity Score for pickle data ----------------------
def quanta_simscore(pickle):
    def quanta_simscore(score): # Function to map similarity score to points based on the new rules
        if score > 0.4:  #Only add a point if the simmialrity score is greater than 0.7
            return 1

    similarity_scores = {} # Empty dictionary 
    for idx, row in pickle.iterrows(): # Iterate over each row
        points = row.apply(quanta_simscore).sum() # Apply the mapping function to each similarity score in the row and sum the points, ignoring NaN
        similarity_scores[idx] = points # Store the total points as the overall similarity score for each idx

    quanta_simscore_df = pd.DataFrame(list(similarity_scores.items()), columns=['idx', 'similarity_score']) # Convert the results to a DataFrame
    return quanta_simscore_df