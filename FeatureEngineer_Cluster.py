import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

'''
df= pd.read_csv('/Volumes/Untitled/Users/dino/Desktop/MSBA/Spring 24/MINING/NBA/NBAMerge_FINAL.csv')

df = df.rename(columns={'salary': 'Salary'})

#Formating the Salary
def format_salary(salary):
    if salary >= 10**6:
        return '${:,.1f}M'.format(salary / 10**6)
    elif salary >= 10**3:
        return '${:,.0f}K'.format(salary / 10**3)
    else:
        return '${:,.0f}'.format(salary)
    
df['Salary Short'] = df['Salary'].apply(format_salary)
###################################   HARD CODING COLUMNS  ################################################
''' DEFENSIVE PLAYERS OF THE YEAR'''

# Define DPOY winners
dpoy_winners = {
    2000: 'Alonzo Mourning',
    2001: 'Dikembe Mutombo',
    2002: 'Ben Wallace',
    2003: 'Ben Wallace',
    2004: 'Ron Artest',
    2005: 'Ben Wallace',
    2006: 'Ben Wallace',
    2007: 'Marcus Camby',
    2008: 'Kevin Garnett',
    2009: 'Dwight Howard',
    2010: 'Dwight Howard',
    2011: 'Dwight Howard',
    2012: 'Tyson Chandler',
    2013: 'Marc Gasol',
    2014: 'Joakim Noah',
    2015: 'Kawhi Leonard',
    2016: 'Kawhi Leonard',
    2017: 'Draymond Green',
    2018: 'Rudy Gobert',
    2019: 'Rudy Gobert',
    2020: 'Giannis Antetokounmpo',
    2021: 'Rudy Gobert',
    2022: 'Marcus Smart',
    2023: 'Jaren Jackson Jr.'
}
# Create DPOY? column
df['DPOY?'] = df.apply(lambda x: 1 if (x['Year'], x['Player']) in dpoy_winners.items() else 0, axis=1)

'''   CHAMPION '''
nba_champions = {
    2000: "LAL",
    2001: "LAL",
    2002: "LAL",
    2003: "SAS",
    2004: "DET",
    2005: "SAS",
    2006: "MIA",
    2007: "SAS",
    2008: "BOS",
    2009: "LAL",
    2010: "LAL",
    2011: "DAL",
    2012: "MIA",
    2013: "MIA",
    2014: "SAS",
    2015: "GSW",
    2016: "CLE",
    2017: "GSW",
    2018: "GSW",
    2019: "TOR",
    2020: "LAL",
    2021: "MIL",
    2022: "GSW",
    2023: "DEN" }
}
df['Champion?'] = df.apply(lambda x: 1 if (x['Year'], x['Tm']) in nba_champions.items() else 0, axis=1)

'''   MVP '''
mvp_winners = {
    2000: 'Shaquille O\'Neal',
    2001: 'Allen Iverson',
    2002: 'Tim Duncan',
    2003: 'Tim Duncan',
    2004: 'Kevin Garnett',
    2005: 'Steve Nash',
    2006: 'Steve Nash',
    2007: 'Dirk Nowitzki',
    2008: 'Kobe Bryant',
    2009: 'LeBron James',
    2010: 'LeBron James',
    2011: 'Derrick Rose',
    2012: 'LeBron James',
    2013: 'LeBron James',
    2014: 'Kevin Durant',
    2015: 'Stephen Curry',
    2016: 'Stephen Curry',
    2017: 'Russell Westbrook',
    2018: 'James Harden',
    2019: 'Giannis Antetokounmpo',
    2020: 'Giannis Antetokounmpo',
    2021: 'Nikola Jokic',
    2022: 'Nikola Jokic',
    2023: 'Joel Embiid'}
}

df['MVP?'] = df.apply(lambda x: 1 if (x['Year'], x['Player']) in mvp_winners.items() else 0, axis=1)


''' ROOKIE OF THE YEAR '''
roy_winners = {
    2000: ['Elton Brand', 'Steve Francis'],
    2000: 'Steve Francis',
    2001: 'Mike Miller',
    2002: 'Pau Gasol',
    2003: 'Amar\'e Stoudemire',
    2004: 'LeBron James',
    2005: 'Emeka Okafor',
    2006: 'Chris Paul',
    2007: 'Brandon Roy',
    2008: 'Kevin Durant',
    2009: 'Derrick Rose',
    2010: 'Tyreke Evans',
    2011: 'Blake Griffin',
    2012: 'Kyrie Irving',
    2013: 'Michael Carter-Williams',
    2014: 'Andrew Wiggins',
    2015: 'Karl-Anthony Towns',
    2016: 'Ben Simmons',
    2017: 'Malcolm Brogdon',
    2018: 'Luka Doncic',
    2019: 'Ja Morant',
    2020: 'LaMelo Ball',
    2021: 'Anthony Edwards',
    2022: 'Scottie Barnes',
    2023: 'Paolo Banchero'}
}
df['ROY?'] = df.apply(lambda x: 1 if (x['Year'], x['Player']) in roy_winners.items() else 0, axis=1)

''' MOST IMPROVED PLAYER'''

mip_winners= {
    2000: 'Jalen Rose',
    2001: 'Tracy McGrady',
    2002: 'Jermaine O\'Neal',
    2003: 'Gilbert Arenas',
    2004: 'Zach Randolph',
    2005: 'Bobby Simmons',
    2006: 'Boris Diaw',
    2007: 'Monta Ellis',
    2008: 'Hedo Türkoğlu',
    2009: 'Danny Granger',
    2010: 'Aaron Brooks',
    2011: 'Kevin Love',
    2012:'Ryan Anderson',
    2013: 'Paul George',
    2014: 'Goran Dragić',
    2015: 'Jimmy Butler',
    2016: 'CJ McCollum',
    2017: 'Giannis Antetokounmpo',
    2018: 'Victor Oladipo',
    2019: 'Pascal Siakam',
    2020: 'Brandon Ingram',
    2021: 'Julius Randle',
    2022: 'Ja Morant',
    2023: 'Lauri Markkanen'}

df['MIP?'] = df.apply(lambda x: 1 if (x['Year'], x['Player']) in mip_winners.items() else 0, axis=1)

###################################################################################
''' FEATURE ENGINEERING OFFENSIVE RATING'''

# Calculate Points Produced
df['Points Produced'] = df['PTS'] + 0.5 * df['AST'] # Assist contribute to half point scored from assisted field goal => adjust

# Calculate Assist Turnover Rate
df['AST/TOV'] = df['AST'] / df['TOV']

# Calculate Offensive Contributtion
df['Off. Effect'] = df['ORB'] +df['STL']- df['TOV'] # FT less efficient than field goals, so coeffiecient of 0.44 estimate points score per FT average in NBA games

# Calculate 
df['Offensive Estinmate'] = df['Points Produced'] +df['Off. Effect']

#Defensive Estimate
df['Defensive Estimate'] = df['STL']*1.3 + df['BLK'] + 0.55*df['DRB'] - 0.44 * df['PF']

pay = df[df['Salary Short'] != '$nan']

# Find the index of the player with the highest 'Defensive Estimate' each year
Huy_DPOY_Pick_index = df.groupby('Year')['Defensive Estimate'].idxmax()

# Create a new DataFrame 'Huy_DPOY_Pick' containing the player with the highest 'Defensive Estimate' each year
Huy_DPOY_Pick = df.loc[Huy_DPOY_Pick_index, ['Year', 'Player', 'Defensive Estimate']]



df.to_csv('NBATotal_HuyStats_v3.csv', index=False)

# Save pay subset to CSV
pay.to_csv('NBASalary_v3.csv', index=False)

########################################################################

df=pd.read_csv('NBATotal_HuyStats_v3.csv')

#####################################   OUR DPOY AND ACTUAL DEPOY ####################################

# Find the index of the player with the highest 'Defensive Estimate' each year
Huy_DPOY_Pick_index = df.groupby('Year')['Defensive Estimate'].idxmax()
# Create a new DataFrame 'Huy_DPOY_Pick' containing the player with the highest 'Defensive Estimate' each year
Huy_DPOY_Pick = df.loc[Huy_DPOY_Pick_index, ['Year', 'Player', 'Defensive Estimate']]

titleus = "Our Defensive Players Each Year"
print(f"\n# {titleus.center(50, '=')}\n\n{Huy_DPOY_Pick}")

#Same Table for ACTUAL DPOY

DPOY_actual= df[df['DPOY?'] == 1][['Year', 'Player', 'Defensive Estimate']].sort_values(by='Year')

titleactual = "ACTUAL Defensive Players Each Year"
print(f"\n# {titleactual.center(50, '=')}\n\n{DPOY_actual}")'''



#####################################   CREATING PCA ####################################
salary= pd.read_csv('NBASalary_v3.csv')
salary.head(15)

# Group by 'Player' and calculate career averages for numerical columns
num_avg_cols = ['Age','G', 'GS', 'MP_', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'eFG%', 
                'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 
                'TOV', 'PF', 'PTS', 'Salary', 'Points Produced', 'AST/TOV', 
                'Off. Effect', 'Offensive Estinmate', 'Defensive Estimate']
num_avg_sal = salary.groupby('Player')[num_avg_cols].mean()

# Group by 'Player' and sum achievements for categorical columns
cat_sum_cols = ['DPOY?', 'Champion?', 'MVP?', 'ROY?', 'MIP?']
cat_sum_df = salary.groupby('Player')[cat_sum_cols].sum()

# Merge the two DataFrames
career_avg = pd.merge(num_avg_sal, cat_sum_df, on='Player')

career_avg.dropna(inplace=True)
career_avg.to_csv('Career_Averages.csv', index=True)

career_avg=pd.read_csv('Career_Averages.csv')
# Display the new DataFrame containing career averages and achievements

############################################################################################################
pca_df= career_avg[['FG%', '3P%', 'FT%', 'TRB','AST/TOV', 'Off. Effect', 'Offensive Estinmate', 'Defensive Estimate', 'Salary', 'DPOY?', 'Champion?', 'MVP?', 'ROY?', 'MIP?']]

variance = pca_df.var()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=variance.index, y=variance.values)
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.xlabel('Column')
plt.ylabel('Variance')
plt.title('Variance of Columns in pca_df')
plt.show()

# Create a scatter plot

pca_df = pca_df.dropna()
##step 1: we normalize the data
##import, initializa, train, transform

from sklearn.preprocessing import StandardScaler #import
scaler=StandardScaler() #initialize
scaler.fit(pca_df) #train, calculating the mean and std deviation

scaled_df=scaler.transform(pca_df)
salary.info()
##step 2: dothe PCA (import, initializa, train, transform)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaled_df)

pca_df=pca.transform(scaled_df) #transform, represent data using new rotated axis

##how much variance/information is captured by the new variables (PC's)
pca.explained_variance_ratio_ # array([0.2851498 , 0.23526245])


#Interpret the PCA

pca_df=pd.DataFrame(pca_df,columns=['PC1','PC2']) #Creating PCA Columns

##Concatinate them together
total_set=pd.concat([pca_df,career_avg],axis=1) #Adding back to DF1 without text data

##Loadings: correlation between the old and new variables
loadings=total_set.corr() #threshold value of 0.5

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 15))
plt.scatter(total_set['PC1'], total_set['PC2'], color='blue', alpha=0.5)

for i in range(len(total_set)):
    plt.text(total_set.loc[i,'PC1'], total_set.loc[i,'PC2'], str(career_avg.loc[i,'Player']), color='black', fontsize=8)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


 #####################################   CREATING CLUSTERS WITH SALARY ####################################
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
########################################## K-MEAN ##########################################
df= pd.read_csv('Career_Averages.csv')
df2 = df[['eFG%','Salary', 'Points Produced', 'AST/TOV','Off. Effect', 'Offensive Estinmate', 'Defensive Estimate', 'DPOY?', 'Champion?', 'MVP?', 'ROY?', 'MIP?']]

# Data scaling
scaler=MinMaxScaler() #initialize
scaler.fit(df2)
scaled_df=scaler.transform(df2)

# Random Guess 4 clusters
km4= KMeans(n_clusters=4)
km4.fit(scaled_df)

## Elbow Mdethod and Siloutetter score
wvc=[]  ## Within clustre variation
sil_scores=[]

for i in range (2,15):
    km= KMeans(n_clusters=i)
    km.fit(scaled_df)
    wvc.append(km.inertia_)
    sil_scores.append(silhouette_score(scaled_df, km.labels_))
    
## Elbow Plot
 
plt.plot(range(2,15), wvc)
plt.xlabel('Num clusters')
plt.ylabel('Within cluster variation') # Plot show best K= 3 cluster

## Sil Score
plt.plot(range(2,15), sil_scores)
plt.xlabel('Num clusters')
plt.ylabel('Score') # Best Score at K= 3 cluster as well

# 3 CLUSTER ACCORDING TO PLOTS
km3= KMeans(n_clusters=3)
km3.fit(scaled_df)

## Elbow Mdethod and Siloutetter score
wvc=[]  ## Within clustre variation
sil_scores=[]

df['label']=km3.labels_

df.head()

# Define a dictionary mapping label values to player tiers
tier_mapping = {0: 'Superstars', 1: 'Bench', 2: 'Role/Key players'}

# Create the 'Player Tier' column by mapping label values to player tiers
df['Player Tier'] = df['label'].map(tier_mapping)


# Scatterplot of data. Interpret graph


import seaborn as sns
import matplotlib.pyplot as plt

    plt.figure(figsize=(13, 13))
    sns.scatterplot(data=df, x='Defensive Estimate', y='Offensive Estinmate', hue='Player Tier', size='Salary', style='Player Tier',palette='dark', sizes=(2, 850))
    
    plt.xlabel('Defensive Estimate')
    plt.ylabel('Offensive Estimate')
    plt.legend(title='Tier Label', bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
    plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(13, 13))
sns.scatterplot(data=df, x='Defensive Estimate', y='Offensive Estinmate', hue='Player Tier', size='Salary', style='Player Tier', palette='dark', sizes=(2, 850))

plt.xlabel('Defensive Estimate')
plt.ylabel('Offensive Estimate')
plt.legend(title='Tier Label', bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside

# Show 'Player' text on each dot for 'Superstars' tier only
for i in range(len(df)):
    if df.loc[i, 'Player Tier'] == 'Superstars':
        plt.text(df.loc[i, 'Defensive Estimate'], df.loc[i, 'Offensive Estinmate'], df.loc[i, 'Player'])

plt.show()

# Groupby  
df.groupby('label').mean()

# Convert DataFrame to CSV
df.to_csv('AvgSalary_Cluster.csv', index=False)


















