import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d
import plotly.graph_objects as go


#----------------Extracting all main data for import (not upsampled) ------------------
tr_data = []  # Empty list to hold TR data
vl_data = []  # Empty list to hold VL data
memorization_score = []  # Empty list to hold memorization scores
data_path = '/Users/ilanastern/Desktop/aiConfidential/data' 
for root, dirs, files in os.walk(data_path):
    if root.endswith('_TR.eval') or root.endswith('_VL.eval') and '_upsample_rand50' not in root: # Check if we are in a TR.eval or VL.eval folder
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
    if '_upsample_rand50' in root and (root.endswith('_TR.eval') or root.endswith('_VL.eval')):
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

def plotter(memscoredf, simscoredf, simscore_name='Data Frame Name'):
    # Ensuring 'idx' columns are the same data type
    memscoredf['idx'] = memscoredf['idx'].astype(str)
    simscoredf['idx'] = simscoredf['idx'].astype(str)
    
    # Merging DataFrames on 'idx'
    merged_df = pd.merge(memscoredf, simscoredf, on='idx', how='inner')
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

    # Setting up the figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # Smaller figure size

    # Plot 1: Memorization Score vs Distribution Score (KDE by Moltype)
    for moltype in moltypes:
        subset = merged_df[merged_df['moltype'] == moltype]
        axs[0].scatter(subset['similarity_score'], subset['Memorization_Score'],
                       color=color_map[moltype], label=f'Moltype {moltype}')
    axs[0].plot(x_fit, y_fit, color='red', label=f'Exp. Decay fit: {params[0]:.2f} * exp(-{params[1]:.2f} * x) + {params[2]:.2f}')
    axs[0].set_xlabel('Distribution Score')
    axs[0].set_ylabel('Memorization Score')
    axs[0].set_yscale('log')
    axs[0].set_title(f'Memorization Score vs Distribution Score ({simscore_name} by Moltype)')
    axs[0].legend()

    # Plot 2: Best Fit Line with Error Bars
    perr = np.sqrt(np.diag(covariance))  # Standard deviation of parameters
    y_err = exp_decay(x_fit, *(params + perr)) - exp_decay(x_fit, *params)
    axs[1].plot(x_fit, y_fit, color='red', label=f'Best Fit: y = {params[0]:.2f} * exp(-{params[1]:.2f} * x) + {params[2]:.2f}')
    axs[1].fill_between(x_fit, y_fit - y_err, y_fit + y_err, color='gray', alpha=0.3, label='Fit Uncertainty')
    axs[1].set_xlabel('Similarity Score')
    axs[1].set_ylabel('Memorization Score')
    axs[1].set_yscale('log')
    axs[1].set_title(f'Memorization Score vs Similarity Score ({simscore_name} with Uncertainty)')
    axs[1].legend()

    # Plot 3: Bubble Chart to Show Density of Points
    density, x_edges, y_edges = np.histogram2d(x, y, bins=50)
    x_center = (x_edges[:-1] + x_edges[1:]) / 2
    y_center = (y_edges[:-1] + y_edges[1:]) / 2
    x_mesh, y_mesh = np.meshgrid(x_center, y_center)
    axs[2].scatter(x_mesh.ravel(), y_mesh.ravel(), s=density.ravel() * 10, alpha=0.6, label='Density of Points')
    axs[2].plot(x_fit, y_fit, color='red', label=f'Fit: y = {params[0]:.2f} * exp(-{params[1]:.2f} * x) + {params[2]:.2f}')
    axs[2].set_xlabel('Similarity Score')
    axs[2].set_ylabel('Memorization Score')
    axs[2].set_title(f'Memorization Score vs Similarity Score ({simscore_name} Bubble Chart)')
    axs[2].legend()

    # Adjusting layout for better spacing
    plt.tight_layout()
    plt.show()

    print(f'The exponential decay equation is: y = {params[0]:.2f} * e^(-{params[1]:.2f} * x) + {params[2]:.2f}')


#-------------------Exponential Decay Function for the pickle data--------------------
def exponential_sim_scores(pickle_df):
    # Constants for the exponential decay function
    A = 10  # Maximum points for the highest score (close to 1)
    B = np.log(A) / (1 - 0.85)  # Solve for B so that a score of 0.85 maps to 1 point

    def map_similarity_to_points(score):
        if pd.isna(score) or score >= 1 or score < 0.0:  # Exclude NaN and out-of-bound values
            return 0
        return A * np.exp(-B * (1 - score))  # Apply exponential decay function
    # Exponetial decay is fit by the equation y = 6.66e^-(1-x)

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

#----------------------Functions to sort for only high performing F1 scores (upsample)-----------------------
def f1above(df, threshold):
    df = df[df['tr_avg_f1'] > threshold] #Return only rows where the TR F1 score is above the threshold
    return df

def f1_above_median_per_moltype(df):
    median_f1 = df.groupby('moltype')['tr_avg_f1'].transform('median') # Calculate the median F1 score for each moltype
    df = df[df['tr_avg_f1'] > median_f1] # Filter rows where the TR F1 score is above the median
    return df

#-----------------------Plotly violin plotter -------------------------------
def plot_f1_density(df):
    fig = go.Figure()  # Initialize plotly figure
    moltypes = df['moltype'].unique()  # Get unique moltypes
    # Loop through each moltype to create the density plot
    for moltype in moltypes:
        subset = df[df['moltype'] == moltype]
        # Create violin plot for tr_avg_f1
        fig.add_trace(go.Violin(
            x=[moltype] * len(subset),  # Group by moltype
            y=subset['tr_avg_f1'],
            side='negative',  # Place the training F1 scores on the left
            line_color='blue',
            name=f"{moltype} tr_avg_f1",
            points=False,
            showlegend=False  ))

        # Create violin plot for vl_avg_f1
        fig.add_trace(go.Violin(
            x=[moltype] * len(subset),
            y=subset['vl_avg_f1'],
            side='positive',  # Place the validation F1 scores on the right
            line_color='green',
            name=f"{moltype} vl_avg_f1",
            points=False,
            showlegend=False ))
        # Add dashed horizontal lines for medians
        median_tr = subset['tr_avg_f1'].median()
        median_vl = subset['vl_avg_f1'].median()
        fig.add_shape(
            type="line",
            x0=moltypes.tolist().index(moltype) - 0.4,  # Align with moltype category
            x1=moltypes.tolist().index(moltype) + 0.4,
            y0=median_tr,
            y1=median_tr,
            line=dict(color="blue", width=2, dash="dash"),
        )
        fig.add_shape(
            type="line",
            x0=moltypes.tolist().index(moltype) - 0.4,
            x1=moltypes.tolist().index(moltype) + 0.4,
            y0=median_vl,
            y1=median_vl,
            line=dict(color="green", width=2, dash="dash"),
        )

    fig.update_layout(
        title="Violin Density Plot for F1 Scores with Medians",
        xaxis=dict(title="Moltype", tickvals=list(range(len(moltypes))), ticktext=moltypes),
        yaxis_title="F1 Score",
        violingap=0.3,
        violinmode='overlay',
        showlegend=False,)
    fig.show()


def plot_memorization_violin(df):
    fig = go.Figure()  # Initialize plotly figure
    moltypes = df['moltype'].unique()  # Get unique moltypes
    # Loop through each moltype to create the violin plot
    for moltype in moltypes:
        subset = df[df['moltype'] == moltype]
        
        fig.add_trace(go.Violin(
            x=[moltype] * len(subset), 
            y=subset['Memorization_Score'],
            line_color='purple',
            name=f"{moltype} Memorization_Score",
            #points='all', 
            showlegend=False))

        # Add dashed horizontal line for median
        median_score = subset['Memorization_Score'].median()
        fig.add_shape(
            type="line",
            x0=moltypes.tolist().index(moltype) - 0.4,  # Align with moltype category
            x1=moltypes.tolist().index(moltype) + 0.4,
            y0=median_score,
            y1=median_score,
            line=dict(color="purple", width=2, dash="dash"),)
    # Update layout for better visualization
    fig.update_layout(
        title="Violin Plot for Memorization Scores by Moltype",
        xaxis=dict(title="Moltype", tickvals=list(range(len(moltypes))), ticktext=moltypes),
        yaxis_title="Memorization Score",
        violingap=0.3,
        violinmode='overlay',
        showlegend=False,)
    fig.show()

#---------------------  Function to normal functions and upsampled functions)---------------------
#Input the dfs that contain the memorization scores
def overlay_upsample_normal(normal_df, upsample_df):
    normal = f1_above_median_per_moltype(normal_df) # Extract high-performing F1 scores
    upsampled = f1_above_median_per_moltype(upsample_df)
    similarity = exponential_sim_scores(pickle) # Calculate similarity scores for the pickle data

    normal.loc[:, 'idx'] = normal['idx'].astype(str)
    upsampled.loc[:, 'idx'] = upsampled['idx'].astype(str)
    similarity.loc[:, 'idx'] = similarity['idx'].astype(str)
    
    # Merging DataFrames on 'idx'
    merged_normal = pd.merge(normal, similarity, on='idx', how='inner')
    merged_normal = merged_normal.replace([np.inf, -np.inf], np.nan).dropna(subset=['similarity_score', 'Memorization_Score'])
    merged_upsampled = pd.merge(upsampled, similarity, on='idx', how='inner')
    merged_upsampled = merged_upsampled.replace([np.inf, -np.inf], np.nan).dropna(subset=['similarity_score', 'Memorization_Score'])

    x_normal = merged_normal['similarity_score']
    y_normal = merged_normal['Memorization_Score']
    x_upsampled = merged_upsampled['similarity_score']
    y_upsampled = merged_upsampled['Memorization_Score']
    
    # Fit exponential decay 
    initial_guess = [1, 1, 1]
    params_normal, covariance_normal = curve_fit(exp_decay, x_normal, y_normal, p0=initial_guess)
    x_fit_normal = np.linspace(min(x_normal), max(x_normal), 500)
    y_fit_normal = exp_decay(x_fit_normal, *params_normal)
    params_upsampled, covariance_upsampled = curve_fit(exp_decay, x_upsampled, y_upsampled, p0=initial_guess)
    x_fit_upsampled = np.linspace(min(x_upsampled), max(x_upsampled), 500)
    y_fit_upsampled = exp_decay(x_fit_upsampled, *params_upsampled)
    
    # Print the equations for the fits
    print(f"Normal Data Fit: y = {params_normal[0]:.5f} * exp(-{params_normal[1]:.5f} * x) + {params_normal[2]:.5f}")
    print(f"Upsampled Data Fit: y = {params_upsampled[0]:.5f} * exp(-{params_upsampled[1]:.5f} * x) + {params_upsampled[2]:.5f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x_normal, y_normal, color='blue', alpha=0.5, label='Normal Data')
    plt.plot(x_fit_normal, y_fit_normal, color='blue', linestyle='--', label='Normal Fit')
    plt.scatter(x_upsampled, y_upsampled, color='orange', alpha=0.5, label='Upsampled Data')
    plt.plot(x_fit_upsampled, y_fit_upsampled, color='orange', linestyle='--', label='Upsampled Fit')
    plt.xlabel('Similarity Score')
    plt.ylabel('Memorization Score')
    plt.yscale('log')
    plt.title('Overlay of Normal and Upsampled Data with Exponential Fits')
    plt.legend()
    plt.show()



#------------------------------Influnce Score Generator and Plotter-----------------------------

def influence_matrix(directory):
    all_contributions = {}  # Dictionary
    for root, _, files in os.walk(directory):  # Loop through each file in the directory
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Loop only relevant files and assign either as training or validation
            if file_name.endswith(".csv") and file_name.startswith("eval_loss_meta"):
                if root.endswith("_TR.eval"):
                    split_type = "train"
                elif root.endswith("_VL.eval"):
                    split_type = "validation"
                else:
                    continue  # Skip all files that arent eval_loss_meta training

                try: #load file in 
                    data = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                avg_f1 = data['f1'].mean() #average f1 score for the file

                for _, row in data.iterrows(): # Contribution to F1 score for each data point
                    datapoint_id = row['idx']
                    f1_score = row['f1']
                    contribution = f1_score - avg_f1

                    if datapoint_id not in all_contributions:
                        all_contributions[datapoint_id] = [] #making sure each data point has a list of contributions
                    all_contributions[datapoint_id].append(contribution)

    unique_datapoints = list(all_contributions.keys()) # list of all datapoints for future plotting
    num_points = len(unique_datapoints)
    pairwise_matrix = np.zeros((num_points, num_points)) # creating pairwise matrix

    for i, dp1 in enumerate(unique_datapoints): # Calculating influnces
        for j, dp2 in enumerate(unique_datapoints):
            if i == j:
                pairwise_matrix[i, j] = 0 # A data point's influence on itself is 0
            else:
                pairwise_matrix[i, j] = np.mean(all_contributions[dp1]) - np.mean(all_contributions[dp2]) # Influence of dp1 on dp2

    pairwise_df = pd.DataFrame(pairwise_matrix,index=unique_datapoints,columns=unique_datapoints) #making matrix into a dataframe
    return pairwise_df

pairwise_influence_df = calculate_pairwise_influence_matrix(data_path)

def plot_influence_vs_similarity(pairwise_influence_df):
    similarity_df = pickle # Load similarity matrix from pickle file, stored in FunctionsAndData.py
    # Convert indices and column names to strings, then extract the portion before the underscore
    pairwise_influence_df.index = pairwise_influence_df.index.astype(str).str.split('_').str[0]
    pairwise_influence_df.columns = pairwise_influence_df.columns.astype(str).str.split('_').str[0]
    similarity_df.index = similarity_df.index.astype(str).str.split('_').str[0]
    similarity_df.columns = similarity_df.columns.astype(str).str.split('_').str[0]

    # Ensure indices and columns match in both DataFrames
    matching_indices = pairwise_influence_df.index.intersection(similarity_df.index)
    matching_columns = pairwise_influence_df.columns.intersection(similarity_df.columns)

    similarity_df = similarity_df.loc[matching_indices, matching_columns] # Filter both matrices to only include matching rows and columns
    pairwise_influence_df = pairwise_influence_df.loc[matching_indices, matching_columns]

    influence_scores = []
    similarity_scores = []

    for idx in matching_indices:
        for col in matching_columns:
            if idx in similarity_df.index and col in similarity_df.columns:
                similarity_score = similarity_df.loc[idx, col]
                influence_score = pairwise_influence_df.loc[idx, col]

                similarity_score = pd.to_numeric(similarity_score, errors='coerce') #errors set to NaN. making sure it's a number
                influence_score = pd.to_numeric(influence_score, errors='coerce')

                if pd.notna(similarity_score) and pd.notna(influence_score): #append if exists
                    similarity_scores.append(similarity_score)
                    influence_scores.append(influence_score)


    if len(influence_scores) == 0 or len(similarity_scores) == 0:  # Ensure valid data
        print("No valid data for plotting. Check your DataFrames or filters.")
        return
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(influence_scores, similarity_scores, alpha=0.5)
    plt.title("Influence Score vs Similarity Score (Matching Rows and Columns)")
    plt.xlabel("Similarity Score")
    plt.ylabel("Influence Score")
    plt.grid(True)
    plt.show()


