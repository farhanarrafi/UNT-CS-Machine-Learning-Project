import datetime as datetime
import pandas as pd
import numpy as np
import os
import csv






# Directory containing the .csv files
directory = 'C:/Users/andre/Documents/.MS/Spring 2024/Machine Learning 5215/Project'

# Initialize an empty DataFrame to store the sums
percentage_volatility = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(directory, filename))
        
        # Calculate the sum of the first ten columns for each row
        df['date'] = pd.to_datetime(df['date'])

        desired_year = 2015

        year_data = df[df['date'].dt.year == desired_year]

        print(year_data)


        print("Number of rows:", year_data.shape[0])


        log_returns = np.log(year_data.close/year_data.close.shift(1)).dropna()


        daily_std = log_returns.std()

        annualized_vol = (daily_std * np.sqrt(252))*100  #Volatility in a year in percentage
                
        # Append the sum as a new row to the sum DataFrame
        percentage_volatility.append(annualized_vol)


vol_df = pd.DataFrame({'Volatility': percentage_volatility})
# Save the sum DataFrame to a new CSV file
vol_df.to_csv('output_file.csv', index=False)


# file_path = 'NVDA_data.csv'

# df = pd.read_csv(file_path)

# print(df)



# # Convert the 'date' column to datetime type
# df['date'] = pd.to_datetime(df['date'])

# desired_year = 2015

# year_data = df[df['date'].dt.year == desired_year]

# print(year_data)


# print("Number of rows:", year_data.shape[0])


# log_returns = np.log(year_data.close/year_data.close.shift(1)).dropna()


# daily_std = log_returns.std()

# annualized_vol = (daily_std * np.sqrt(252))*100  #Volatility in a year in percentage
