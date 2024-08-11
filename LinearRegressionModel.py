import pandas as pd
import rfpimp
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
df = pd.read_csv('/Volumes/Untitled/Users/dino/Desktop/MSBA/Spring 24/MINING/NBA/AvgSalary_Cluster.csv')
df.head(10)

'''
df.dropna(inplace=True)
df = df[(df['Year'] >= 2015) & (df['Year'] <= 2020)]

##############################  Get Average  #############################

# Group by 'Player' and calculate career averages for numerical columns
num_avg_cols = ['Age','G', 'GS', 'MP_', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'eFG%', 
                'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 
                'TOV', 'PF', 'PTS', 'Salary', 'Points Produced', 'AST/TOV', 
                'Off. Effect', 'Offensive Estinmate', 'Defensive Estimate']
num_avg_sal = df.groupby('Player')[num_avg_cols].mean()

# Group by 'Player' and sum achievements for categorical columns
cat_sum_cols = ['DPOY?', 'Champion?', 'MVP?', 'ROY?', 'MIP?']
cat_sum_df = df.groupby('Player')[cat_sum_cols].sum()

# Merge the two DataFrames
df = pd.merge(num_avg_sal, cat_sum_df, on='Player')'''
################################ RANDOM FOREST ################################

features = [ 'G', 'GS', 'MP_', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Salary', 'Points Produced', 'AST/TOV', 'Off. Effect', 'Offensive Estinmate', 'Defensive Estimate', 'DPOY?', 'Champion?', 'MVP?', 'ROY?', 'MIP?']

######################################## Train/test split #########################################

df_train, df_test = train_test_split(df, test_size=0.10)
df_train = df_train[features]
df_test = df_test[features]

X_train, y_train = df_train.drop('Salary',axis=1), df_train['Salary']
X_test, y_test = df_test.drop('Salary',axis=1), df_test['Salary']

################################################ Train #############################################

rf = RandomForestRegressor(n_estimators=800, n_jobs=-1)
rf.fit(X_train, y_train)

############################### Permutation feature importance #####################################

imp = rfpimp.importances(rf, X_test, y_test)
y_pred = rf.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

print("R2 Score:", r2)
print("Mean Absolute Error:", mae)


############################################## Plot ################################################

fig, ax = plt.subplots(figsize=(9,7))

# Sort the DataFrame by importance score in descending order
imp_sorted = imp.sort_values(by='Importance', ascending=False)

# Define colormap
cmap = plt.cm.get_cmap('coolwarm')

# Plot horizontal bar chart with gradient color
bars = ax.barh(imp_sorted.index, imp_sorted['Importance'], height=0.8, 
               color=cmap(imp_sorted['Importance']/imp_sorted['Importance'].max()))

# Add labels and title
ax.set_xlabel('Importance score')
ax.set_title('What Statistic Matters Most When Predicting Salary?', fontsize=12, weight='bold')

# Add a label for color scale
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=None, cmap=cmap), ax=ax)
cbar.set_label('Color Scale')

# Invert y-axis
plt.gca().invert_yaxis()

# Add text annotation
for bar in bars:
    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
            f'{bar.get_width():.2f}', va='center')

# Adjust layout
fig.tight_layout()

plt.show()
################################################ Linear Model ########################################
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# Assuming X and y are already defined
X = df['Offensive Estinmate'].values.reshape(-1,1)  # Include multiple features
y = df['Salary'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Create and fit the linear regression model on the training data
ols = linear_model.LinearRegression()
model = ols.fit(X_train, y_train)
response = model.predict(X)
# Make predictions on the testing data
y_pred_test = model.predict(X_test)

# Calculate RMSE for testing data
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
print("RMSE Testing Data:", rmse_test)

# Make predictions on the training data
y_pred_train = model.predict(X_train)

# Calculate RMSE for training data
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
print("RMSE Training Data:", rmse_train)

# Calculate R-squared
r2 = model.score(X_test, y_test)
print("R-squared:", r2)

############################################## Plot ################################################

plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(9, 8))

ax.plot(X, response, color='k', label='Regression model')
ax.scatter(X, y, edgecolor='k', facecolor='purple', alpha=0.7, label='Sample data')
ax.set_ylabel('Salary x 10M ($)', fontsize=14)
ax.set_xlabel('Offensive Estimates', fontsize=14)
ax.text(0.8, 0.1,"r2=0.59358",fontsize=20, ha='center', va='center',
         transform=ax.transAxes, color='black', alpha=0.5)
ax.legend(facecolor='white', fontsize=11)
ax.set_title('Linear Regression of Salary vs Offensive Estimates', fontsize=18)

fig.tight_layout()

######################   n 3D Linear Model #####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D


X = df[['Offensive Estinmate', 'Defensive Estimate']].values.reshape(-1,2)
Y = df['Salary']

######################## Prepare model data point for visualization ###############################

x = X[:, 0]
y = X[:, 1]
z = Y


# For x-axis range (Offensive Estimate)
offensive_estimate_range = df['Offensive Estinmate'].min(), df['Offensive Estinmate'].max()

# For y-axis range (Defensive Estimate)
defensive_estimate_range = df['Defensive Estimate'].min(), df['Defensive Estimate'].max()

# For z-axis range (Salary)
salary_range = df['Salary'].min(), df['Salary'].max()


# Adjusted ranges
x_min, x_max = offensive_estimate_range
y_min, y_max = defensive_estimate_range

# Use salary range for z-axis
z_min, z_max = salary_range

# Generate prediction grid using adjusted ranges
x_pred = np.linspace(x_min, x_max, 30)
y_pred = np.linspace(y_min, y_max, 30)
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

################################################ Train #############################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import linear_model

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# Train the linear regression model on the training data
ols = linear_model.LinearRegression()
model = ols.fit(X_train, y_train)
predicted = model.predict(model_viz)
# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)

r2 = r2_score(y_test, y_pred)
print("R²:", r2)
############################################## Plot ################################################

####

####
df = df.dropna()


fig = plt.figure(figsize=(15, 10))  

ax1 = fig.add_subplot(111, projection='3d', aspect='auto')

scatter = ax1.scatter(x, y, z, c=df['label'], zorder=15, marker='o', alpha=1)
ax1.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=15, edgecolor='black')

ax1.set_xlabel('Offensive Est.', fontsize=15)
ax1.set_ylabel('Defensive Est.', fontsize=15)
ax1.set_zlabel('Salary x 10M ($)', fontsize=15)

ax1.locator_params(nbins=4, axis='x')
ax1.locator_params(nbins=5, axis='x')

ax1.view_init(elev=12, azim=120)

fig.suptitle('$R^2 = %.2f$' % r2, fontsize=50, fontweight='bold')

fig.colorbar(scatter, label='Cluster Group')  # Add colorbar

fig.tight_layout()
plt.show()
##
for ii in np.arange(0, 360, 1):
    ax1.view_init(elev=10, azim=ii)
    fig.savefig('/Volumes/Untitled/Users/dino/Desktop/MSBA/Spring 24/MINING/NBA/3D/gif_image%d.png' % ii)

    
# Close the figure
plt.close(fig)

df2=pd.read_csv('/Volumes/Untitled/Users/dino/Desktop/MSBA/Spring 24/MINING/NBA/AvgSalary_Cluster.csv')
cluster_averages = df2.groupby('Player Tier')[["PTS", "Salary", "Offensive Estinmate", "Defensive Estimate"]].mean()


cluster_tiers = {
    "Cluster 1": "Bench Player",
    "Cluster 2": "Role Player",
    "Cluster 3": "Superstar"
}

# Map cluster names to tiers based on the 'Player Tier' column
df2['Tier'] = df2['Player Tier'].map(cluster_tiers)

# Calculate cluster averages for each cluster
bench_player_avg = df2[df2['Player Tier'] == 'Bench Player'][["PTS", "Salary", "Offensive Estimate", "Defensive Estimate"]].mean()
role_player_avg = df2[df2['Player Tier'] == 'Role Player'][["PTS", "Salary", "Offensive Estimate", "Defensive Estimate"]].mean()
superstar_avg = df2[df2['Player Tier'] == 'Superstar'][["PTS", "Salary", "Offensive Estimate", "Defensive Estimate"]].mean()








