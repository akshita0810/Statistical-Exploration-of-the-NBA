import seaborn as sns
import pandas as pd

# Load CSV files one by one and assign them to DataFrames
nba2000 = pd.read_csv('2000.csv', encoding='latin1')
nba2001 = pd.read_csv('2001.csv', encoding='latin1')
nba2002 = pd.read_csv('2002.csv', encoding='latin1')
nba2003 = pd.read_csv('2003.csv', encoding='latin1')
nba2004 = pd.read_csv('2004.csv', encoding='latin1')
nba2005 = pd.read_csv('2005.csv', encoding='latin1')
nba2006 = pd.read_csv('2006.csv', encoding='latin1')
nba2007 = pd.read_csv('2007.csv', encoding='latin1')
nba2008 = pd.read_csv('2008.csv', encoding='latin1')
nba2009 = pd.read_csv('2009.csv', encoding='latin1')
nba2010 = pd.read_csv('2010.csv', encoding='latin1')
nba2011 = pd.read_csv('2011.csv', encoding='latin1')
nba2012 = pd.read_csv('2012.csv', encoding='latin1')
nba2013 = pd.read_csv('2013.csv', encoding='latin1')
nba2014 = pd.read_csv('2014.csv', encoding='latin1')
nba2015 = pd.read_csv('2015.csv', encoding='latin1')
nba2016 = pd.read_csv('2016.csv', encoding='latin1')
nba2017 = pd.read_csv('2017.csv', encoding='latin1')
nba2018 = pd.read_csv('2018.csv', encoding='latin1')
nba2019 = pd.read_csv('2019.csv', encoding='latin1')
nba2020 = pd.read_csv('2020.csv', encoding='latin1')
nba2021 = pd.read_csv('2021.csv', encoding='latin1')
nba2022 = pd.read_csv('2022.csv', encoding='latin1')
nba2023 = pd.read_csv('2023.csv', encoding='latin1')
nba2024 = pd.read_csv('2024.csv', encoding='latin1')

#Create Year Column
nba2000['Year'] = 2000
nba2001['Year'] = 2001
nba2002['Year'] = 2002
nba2003['Year'] = 2003
nba2004['Year'] = 2004
nba2005['Year'] = 2005
nba2006['Year'] = 2006
nba2007['Year'] = 2007
nba2008['Year'] = 2008
nba2009['Year'] = 2009
nba2010['Year'] = 2010
nba2011['Year'] = 2011
nba2012['Year'] = 2012

nba2013['Year'] = 2013
nba2014['Year'] = 2014
nba2015['Year'] = 2015
nba2016['Year'] = 2016
nba2017['Year'] = 2017
nba2018['Year'] = 2018
nba2019['Year'] = 2019
nba2020['Year'] = 2020
nba2021['Year'] = 2021
nba2022['Year'] = 2022
nba2023['Year'] = 2023
nba2024['Year'] = 2024

nba_2000_2024 = pd.concat([nba2000, nba2001, nba2002, nba2003, nba2004, nba2005, nba2006, nba2007, nba2008, nba2009, nba2010, nba2011, nba2012,nba2013, nba2014, nba2015,nba2016,nba2017, nba2018,nba2019,nba2020,nba2021,nba2022,nba2023,nba2024], ignore_index=True)

nba_2000_2024 = nba_2000_2024[['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'GS', 'MP_', 'FG', 'FGA',
       'FG%', '3P', '3PA', '3P%','eFG%', 'FT', 'FTA',
       'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']]

nba_2000_2024.to_excel('NBA_2000_2024.xlsx', index=False)
