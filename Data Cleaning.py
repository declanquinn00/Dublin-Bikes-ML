import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Read CSV
#df1 = pd.read_csv("2019Q1.csv", delimiter=',')
#df2 = pd.read_csv("2019Q2.csv", delimiter=',')
#df3 = pd.read_csv("2019Q3.csv", delimiter=',')
#df4 = pd.read_csv("2019Q4.csv", delimiter=',')

#data2019 = pd.concat([df1,df2,df3,df4])
#del data2019['LAST UPDATED']
#del data2019['STATUS']
#del data2019['LATITUDE']
#del data2019['LONGITUDE']
#del data2019['ADDRESS']
#del data2019['NAME']
#data2019.to_csv('2019Full.csv')
'''
KEEP ALL OF ABOVE FOR PREPROCESSING!!!
'''

df2019 = pd.read_csv("2019Full.csv")

new_df2019 = df2019[["STATION ID", "TIME", "AVAILABLE BIKE STANDS", "AVAILABLE BIKES"]]

new_df2019['TIME'] = pd.to_datetime(new_df2019['TIME'], format='%Y-%m-%d %H:%M:%S')




# Get weekday and date
new_df2019['DAY'] = new_df2019['TIME'].dt.day_name()
new_df2019['DATE'] = new_df2019['TIME'].dt.date

#remove conecutive duplicate values
#https://stackoverflow.com/questions/19463985/pandas-drop-consecutive-duplicates
df = new_df2019[new_df2019["AVAILABLE BIKES"] != new_df2019["AVAILABLE BIKES"].shift(-1)].dropna()
df = df.reset_index(drop=True)

date_pairs = {}
unique_dates = df['DATE'].unique()

for date in unique_dates:
    temp_df = df[(df['DATE'] == date)]
    diff = 0
    for index in range(1,len(temp_df)):
        last_val = temp_df["AVAILABLE BIKES"].iloc[index-1]
        current_val = temp_df["AVAILABLE BIKES"].iloc[index]
        if temp_df["STATION ID"].iloc[index-1] == temp_df["STATION ID"].iloc[index]:
            diff = diff + abs(last_val - current_val)
    # avoid double counting bikes taken and returned as separate journeys
    diff = diff/2
    date_pairs[date] = diff

# Create dataframe of dict values
df2019_data = pd.DataFrame(list(date_pairs.items()), columns=['Date', 'Bike Usage'])
df2019_data.to_csv('2019data.csv')





# 2020 Data SAVED IN 2022data.csv NO NEED TO RERUN


# Read CSV
df1 = pd.read_csv("2020Q1.csv", delimiter=',')
df2 = pd.read_csv("2020Q2.csv", delimiter=',')
df3 = pd.read_csv("2020Q3.csv", delimiter=',')
df4 = pd.read_csv("2020Q4.csv", delimiter=',')

data2020 = pd.concat([df1,df2,df3,df4])
del data2020['LAST UPDATED']
del data2020['STATUS']
del data2020['LATITUDE']
del data2020['LONGITUDE']
del data2020['ADDRESS']
del data2020['NAME']
data2020.to_csv('2020Full.csv')
'''
KEEP ALL OF ABOVE FOR PREPROCESSING!!!
'''
df2020 = pd.read_csv("2020Full.csv")

new_df2020 = df2020[["STATION ID", "TIME", "AVAILABLE BIKE STANDS", "AVAILABLE BIKES"]]

new_df2020['TIME'] = pd.to_datetime(new_df2020['TIME'], format='%Y-%m-%d %H:%M:%S')


# Get weekday and date
new_df2020['DAY'] = new_df2020['TIME'].dt.day_name()
new_df2020['DATE'] = new_df2020['TIME'].dt.date

#remove conecutive duplicate values
#https://stackoverflow.com/questions/19463985/pandas-drop-consecutive-duplicates
df = new_df2020[new_df2020["AVAILABLE BIKES"] != new_df2020["AVAILABLE BIKES"].shift(-1)].dropna()
df = df.reset_index(drop=True)

date_pairs = {}
unique_dates = df['DATE'].unique()

for date in unique_dates:
    temp_df = df[(df['DATE'] == date)]
    diff = 0
    for index in range(1,len(temp_df)):
        last_val = temp_df["AVAILABLE BIKES"].iloc[index-1]
        current_val = temp_df["AVAILABLE BIKES"].iloc[index]
        if temp_df["STATION ID"].iloc[index-1] == temp_df["STATION ID"].iloc[index]:
            diff = diff + abs(last_val - current_val)
    # avoid double counting bikes taken and returned as separate journeys
    diff = diff/2
    date_pairs[date] = diff

# Create dataframe of dict values
df2020_data = pd.DataFrame(list(date_pairs.items()), columns=['Date', 'Bike Usage'])
df2020_data.to_csv('2020data.csv')






# 2021 SAVED IN 2021data.csv NO NEED TO RERUN

# Read CSV
#df1 = pd.read_csv("2021Q1.csv", delimiter=',')
#df2 = pd.read_csv("2021Q2.csv", delimiter=',')
#df3 = pd.read_csv("2021Q3.csv", delimiter=',')
#df4 = pd.read_csv("2021Q4.csv", delimiter=',')

#data2021 = pd.concat([df1,df2,df3,df4])
#del data2021['LAST UPDATED']
#del data2021['STATUS']
#del data2021['LATITUDE']
#del data2021['LONGITUDE']
#del data2021['ADDRESS']
#del data2021['NAME']
#data2021.to_csv('2021Full.csv')
'''
KEEP ALL OF ABOVE FOR PREPROCESSING!!!
'''

df2021 = pd.read_csv("2021Full.csv")

new_df2021 = df2021[["STATION ID", "TIME", "AVAILABLE BIKE STANDS", "AVAILABLE BIKES"]]

new_df2021['TIME'] = pd.to_datetime(new_df2021['TIME'], format='%Y-%m-%d %H:%M:%S')


# Get weekday and date
new_df2021['DAY'] = new_df2021['TIME'].dt.day_name()
new_df2021['DATE'] = new_df2021['TIME'].dt.date

#remove conecutive duplicate values
#https://stackoverflow.com/questions/19463985/pandas-drop-consecutive-duplicates
df = new_df2021[new_df2021["AVAILABLE BIKES"] != new_df2021["AVAILABLE BIKES"].shift(-1)].dropna()
df = df.reset_index(drop=True)

date_pairs = {}
unique_dates = df['DATE'].unique()

for date in unique_dates:
    temp_df = df[(df['DATE'] == date)]
    diff = 0
    for index in range(1,len(temp_df)):
        last_val = temp_df["AVAILABLE BIKES"].iloc[index-1]
        current_val = temp_df["AVAILABLE BIKES"].iloc[index]
        if temp_df["STATION ID"].iloc[index-1] == temp_df["STATION ID"].iloc[index]:
            diff = diff + abs(last_val - current_val)
    # avoid double counting bikes taken and returned as separate journeys
    diff = diff/2
    date_pairs[date] = diff

# Create dataframe of dict values
df2021_data = pd.DataFrame(list(date_pairs.items()), columns=['Date', 'Bike Usage'])
df2021_data.to_csv('2021data.csv')






# 2022 SAVED IN 2022data.csv NO NEED TO RERUN

# Read CSV
#df1 = pd.read_csv("2022Jan.csv", delimiter=',')
#df2 = pd.read_csv("2022Feb.csv", delimiter=',')
#df3 = pd.read_csv("2022Mar.csv", delimiter=',')
#df4 = pd.read_csv("2022Apr.csv", delimiter=',')
#df5 = pd.read_csv("2022May.csv", delimiter=',')
#df6 = pd.read_csv("2022Jun.csv", delimiter=',')
#df7 = pd.read_csv("2022Jul.csv", delimiter=',')
#df8 = pd.read_csv("2022Aug.csv", delimiter=',')
#df9 = pd.read_csv("2022Sept.csv", delimiter=',')
#df10 = pd.read_csv("2022Oct.csv", delimiter=',')
#df11 = pd.read_csv("2022Nov.csv", delimiter=',')
#df12 = pd.read_csv("2022Dec.csv", delimiter=',')

#data2022 = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12])
#del data2022['LAST UPDATED']
#del data2022['STATUS']
#del data2022['LATITUDE']
#del data2022['LONGITUDE']
#del data2022['ADDRESS']
#del data2022['NAME']
#data2022.to_csv('2022Full.csv')
'''
KEEP ALL OF ABOVE FOR PREPROCESSING!!!
'''

df2022 = pd.read_csv("2022Full.csv")
#df2022 = pd.read_csv("2022debug.csv")

new_df2022 = df2022[["STATION ID", "TIME", "AVAILABLE_BIKE_STANDS", "AVAILABLE_BIKES"]]

new_df2022['TIME'] = pd.to_datetime(new_df2022['TIME'], format='%Y-%m-%d %H:%M:%S')


# Get weekday and date
new_df2022['DAY'] = new_df2022['TIME'].dt.day_name()
new_df2022['DATE'] = new_df2022['TIME'].dt.date

#remove conecutive duplicate values
#https://stackoverflow.com/questions/19463985/pandas-drop-consecutive-duplicates
df = new_df2022[new_df2022["AVAILABLE_BIKES"] != new_df2022["AVAILABLE_BIKES"].shift(-1)].dropna()
df = df.reset_index(drop=True)




date_pairs = {}
unique_dates = df['DATE'].unique()
unique_times = df['TIME'].unique()

# same as before but we sort
for date in unique_dates:
    temp_df = df[(df['DATE'] == date)]
    temp_df = temp_df.sort_values(['STATION ID', 'TIME'])
    # Remove duplicates
    temp_df = temp_df.reset_index(drop=True)
    temp_df = temp_df[temp_df["AVAILABLE_BIKES"] != temp_df["AVAILABLE_BIKES"].shift(-1)].dropna()
    temp_df = temp_df.reset_index(drop=True)
    diff = 0
    for index in range(1,len(temp_df)):
        last_val = temp_df["AVAILABLE_BIKES"].iloc[index-1]
        current_val = temp_df["AVAILABLE_BIKES"].iloc[index]
        # If last station ID == Station ID
        if temp_df["STATION ID"].iloc[index-1] == temp_df["STATION ID"].iloc[index]:
            diff = diff + abs(last_val - current_val)
    # avoid double counting bikes taken and returned as separate journeys
    diff = diff/2
    date_pairs[date] = diff

# Create dataframe of dict values
df2022_data = pd.DataFrame(list(date_pairs.items()), columns=['Date', 'Bike Usage'])
df2022_data.to_csv('2022data.csv')


# Concatenate into one final dataset

df1 = pd.read_csv("2019data.csv", delimiter=',')
df2 = pd.read_csv("2020data.csv", delimiter=',')
df3 = pd.read_csv("2021data.csv", delimiter=',')
df4 = pd.read_csv("2022data.csv", delimiter=',')

bike_usage_total = pd.concat([df1,df2,df3,df4])
bike_usage_total = bike_usage_total.reset_index(drop=True)
bike_usage_total.to_csv('full_cleaned_data_merged.csv')


# Plot dataset
# Read CSV
df = pd.read_csv("full_cleaned_data_merged.csv")
# Drop unnamed columns
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
# Drop duplicate dates
df = df.drop_duplicates(subset='Date')
print(df.head())
# Drop extreme outlier values
df = df.drop(df[df['Bike Usage'] < 100.0].index)
df = df.drop(df[df['Bike Usage'] > 9000.0].index)
df = df.reset_index(drop=True)
# Pandemic times
pand_start = pd.to_datetime("27-03-2020")
pand_end = pd.to_datetime("28-01-2022")

# add day of the week feature
weekdays = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

# For each value calculate weekdays value
for index in range(0, len(df)):
    datetimeString = df['Date'].iloc[index]
    datetimeObject = datetime.strptime(datetimeString, '%Y-%m-%d')
    day = weekdays.get(datetimeObject.strftime('%A'))
    df.at[index, 'Day'] = int(day)


# Calculate pandemic values
df['Date'] = pd.to_datetime(df['Date'])
pre_pandemic_df = df[df['Date'] < pand_start]
pandemic_df = df[(df['Date'] < pand_end) & (df['Date'] >= pand_start)]
post_pandemic_df = df[df['Date'] >= pand_end]

pre_pandemic_df.to_csv('PrePandemicdata.csv')
pandemic_df.to_csv('Pandemicdata.csv')
post_pandemic_df.to_csv('PostPandemicdata.csv')

print('Done')