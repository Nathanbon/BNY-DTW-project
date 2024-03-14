# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:34:07 2024

@author: nathan Bonneau
"""
"""
# Definition of DTW

## Importing libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf              # Importation for using Yahoo finance
import datetime
import pandas as pd
import seaborn as sns
import csv
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from IPython.display import clear_output

"""## Function to load data from Yahoo Finance"""

def if_data(pair) :
  sg = pair + "=X"
  data = yf.Ticker(sg)
  start_datetime = datetime.datetime(2024, 1, 22, 12, 00)
  end_datetime = start_datetime + datetime.timedelta(hours=30) #takes data per minute for 2 hours
  dataDF= yf.download(tickers= sg, start=start_datetime, end=end_datetime, interval='1h')
  array10 = dataDF.index.to_numpy()
  array1 = dataDF['Close'].to_numpy()  # Collect prices at closing
  return array1

array1 = if_data("GBPUSD")
array2 = if_data("EURUSD")

"""
##Check the correct data download"""

indices = range(len(array1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(indices, array1)
ax1.set_xlabel("Time")
ax1.set_ylabel("Closing price (GBPUSD)")
ax1.set_title("Chart for GBPUSD")

ax2.plot(indices, array2)
ax2.set_xlabel("Time")
ax2.set_ylabel("Closing price (EURUSD)")
ax2.set_title("Chart for GBPUSD")

plt.tight_layout()


plt.show()

"""##Define the normalize function
The objective of standardization: to put on a common scale, to avoid the impact of outliers.
"""

def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val) #min-max normalization
    return normalized_array

#Example to better visualize

array1 = [10,40,30,60,80,90,100,50,40,60,80,90,100,90,90,100,90,110,150,160,140,130]
array2 = [10,15,20,30,15,41,33,62,89,91,101,55,49,69,80,90,110,100,92,109,91,120,160,150,150,140,120,120]

# Normalization of our two times series
normalized_array1 = normalize(array1)
normalized_array2 = normalize(array2)

plt.figure(figsize=(10, 4))

# First graph = time series before normalization
plt.subplot(1, 2, 1)
plt.plot(array1, label='Array 1')
plt.plot(array2, label='Array 2')
plt.title('Before normalization')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Second graph = normalized time series
plt.subplot(1, 2, 2)
plt.plot(normalized_array1, label='Array 1 normalized')
plt.plot(normalized_array2, label='Array 2 normalized')
plt.title('After normalization')
plt.xlabel('Time')
plt.ylabel('Normalized value')
plt.legend()

plt.tight_layout()
plt.show()

"""## Construction of the cost matrix"""

#Cost matrix which will have all the manhattan distance between our points
def cost_matrix_compute(a,b):
  cost_matrix = np.zeros((len(a), len(b)))
  for i in range(len(a)):
      for j in range(len(b)):
          cost_matrix[i, j] = abs(a[i] - b[j]) # Manhattan distance
  # Cost_matrix has a dimension of len(a)*len(b)
# We now create a matrix which will have the accumulated distance
  acc_cost_matrix = np.zeros((len(a), len(b)))
  acc_cost_matrix[0, 0] = cost_matrix[0, 0]
  for i in range(1, len(a)):
      acc_cost_matrix[i, 0] = cost_matrix[i, 0] + acc_cost_matrix[i-1, 0]
  for j in range(1, len(b)):
      acc_cost_matrix[0, j] = cost_matrix[0, j] + acc_cost_matrix[0, j-1]
  for i in range(1, len(a)):
      for j in range(1, len(b)):
          acc_cost_matrix[i, j] = cost_matrix[i, j] + min(acc_cost_matrix[i-1, j], acc_cost_matrix[i, j-1], acc_cost_matrix[i-1, j-1])

  # The idea is that acc_cost_matrix[i,j] will contain the minimum cumulative cost to reach point a[i] from point b[j]

  # path is the variable will contain the smallest path.
  # We start from the top right of the cost matrix and we look at the box [i-1,j], [i-1,j-1] and [i,j-1] or the one with the smallest value and repeat the operation

  path = [(len(a)-1, len(b)-1)]
  i = len(a) - 1
  j = len(b) - 1
  while i > 0 or j > 0:
      if i == 0: # In this situation we no longer have a choice
          j -= 1
      elif j == 0: # In this situation we no longer have a choice
          i -= 1
      else:
          min_cost = min(acc_cost_matrix[i-1, j], acc_cost_matrix[i, j-1], acc_cost_matrix[i-1, j-1])
          if min_cost == acc_cost_matrix[i-1, j]:
              i -= 1
          elif min_cost == acc_cost_matrix[i, j-1]:
              j -= 1
          else:
              i -= 1
              j -= 1
      path.append((i, j)) # We add the couple (i,j) to our future shortest path
  path.reverse() # As we started from the end, we must return our path
    # Our objective is to deform a so that it corresponds to the time series b,
  # We check that our path is the same length as b

  while(len(path)!=len(b)):
    new1 = [a[p[0]] for p in path[1:]]
    result1 = sum(abs(b[i] - new1[i]) for i in range(len(b)))
    new2 = [a[p[0]] for p in path[:-1]]
    result2 = sum(abs(b[i] - new2[i]) for i in range(len(b)))
    if result1 > result2 :
      path = path[:-1]
    else :
      path = path[1:]
  # We shorten our distorted time series by removing the beginning or the end, choosing the best option each time
  return path, cost_matrix, acc_cost_matrix


a = normalized_array1
b = normalized_array2 # Our two new times series
path, cost_matrix, acc_cost_matrix = cost_matrix_compute(a,b)


deformed_a = [a[p[0]] for p in path[:]]

# We plot the result: a deformed so that it corresponds to b
plt.plot(a, label='Time serie a')
plt.plot(b, label='Time serie b')
plt.plot(deformed_a, 'r--', label='Distorted version of a')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('DTW')
plt.legend()
plt.show()

"""## Plot the cost matrix and trace the shortest path in red"""

x = [p[1] for p in path]
y = [p[0] for p in path]
plt.imshow(cost_matrix, cmap='hot', origin='lower')
plt.colorbar(label='Coût')
plt.title('Cost Matrix of DTW')
plt.xlabel('Index b')
plt.ylabel('Index a')
plt.plot(x, y, 'r--', linewidth=2, label='Chemin')
plt.legend()
plt.show()

"""## RMSE
Calculation of the squared error between time series b and deformed_a

*   Close to 0: good phase matching
*   Close to 1: poor phase matching

"""

difference = b - deformed_a
squared_difference = difference ** 2
rmse = np.sqrt(np.mean(squared_difference))
print("RMSE:", rmse)

paires_devises = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "EURAUD"]
df = pd.DataFrame(index=paires_devises, columns=paires_devises)
matrix = np.zeros(( 12, 12))
for ligne in df.index:
    for colonne in df.columns:
        array1 = if_data(ligne)
        array2 = if_data(colonne)
        a = normalize(array1)
        b = normalize(array2)
        path, cost_matrix, acc_cost_matrix = cost_matrix_compute(a, b)
        deformed_a = [a[p[0]] for p in path[:]]
        df.loc[ligne, colonne] = np.sqrt(np.mean((deformed_a - b) ** 2))

"""## DTW matrix RMSE between different forex indices"""

import seaborn as sns
import matplotlib.pyplot as plt

df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(0)

plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap="Greens_r", cbar=False)
cax = plt.gcf().axes[-1]
cax.yaxis.set_label_position("right")
cax.yaxis.set_ticks_position("left")
plt.ylabel("Paires déformées")

# The greener the box, the higher the correlation
plt.show()



"""# Trading strategy on GBPUSD and EURUSD - 

## Importation of data with Yahoo Finance
"""

def yfv2(pair, i, years, months, day, h, min) :
  sg = pair + "=X"
  data = yf.Ticker(sg)
  start_datetime = datetime.datetime(years, months,day, h, min)
  end_datetime = start_datetime + datetime.timedelta(minutes=i)
  dataDF = yf.download(tickers= sg, start=start_datetime, end=end_datetime, interval='1m')
  array10 = dataDF.index.to_numpy()  # Time
  array1 = dataDF['Close'].to_numpy()  # Price
  return array1

"""## Importation of data with EXCEL"""

def get_day_close_pricesGBPUSD(year, month, day):
    filename = "C:/Users/natha/Desktop/EURUSD GBPUSD/1min/gbpusd.csv" # Add GBPUSD_M5 to do with data every fifteen minutes
    close_prices = []

    with open(filename, "r") as file:
        csv_reader = csv.reader(file, delimiter="\t")
        next(csv_reader)  # Skip the first line

        for row in csv_reader:
            
            date_time = row[0].split()
            date = date_time[0]
            time = date_time[1]

            # Check if the date matches the input parameters
            if date == f"{year}-{month:02d}-{day:02d}":
                close_price = float(date_time[5])  # Extract the Close price from the fifth column
                close_prices.append(close_price)

    return close_prices

def get_day_close_pricesEURUSD(year, month, day):
    filename = "C:/Users/natha/Desktop/EURUSD GBPUSD/1min/eurusd-data.csv" # Add GBPUSD_M5 to do with data every fifteen minutes
    close_prices = []

    with open(filename, "r") as file:
        csv_reader = csv.reader(file, delimiter=";")
        next(csv_reader)  # Skip the first line

        for row in csv_reader:
            date_time = row[0].split()
            date = date_time[0]
            time = date_time[1]
            # Check if the date matches the input parameters
            if date == f"{day:02d}/{month:02d}/{year}":
                close_price = float(row[1].replace(',', '.'))  # Extract the Close price from the fifth column
                close_prices.append(close_price)

    return close_prices

"""## Exporting results to an excel file"""

def save_styled_dataframe_to_excel(styled_dataframe, merged_dataframe, filename, declenchement, decalage, nbpointsDTW, varMin, varMax,windows1):
    # Check if the file exists
    file_exists = False
    try:
        workbook = openpyxl.load_workbook(filename)
        file_exists = True
    except FileNotFoundError:
        workbook = openpyxl.Workbook()

    # Select the active sheet
    sheet = workbook.active

    # Convert the styled dataframe to a regular dataframe
    dataframe = styled_dataframe.data

    # Append a line of '/' as a separator if the file already exists
    if file_exists:
        separator_row = ['/' for _ in range(len(dataframe.columns))]
        sheet.append(separator_row)
    sheet.append(['Parameters (declenchement, decalage, nbpointsDTW, varMin, varMax, windows):', declenchement, decalage, nbpointsDTW, varMin, varMax,windows1])
    # Append the dataframe to the sheet
    for row in dataframe_to_rows(dataframe, index=False, header=True):
        sheet.append(row)

    # Count the 'win' values
    counts = merged_dataframe['Result'].value_counts()
    win_count = counts.get('win', 0)
    loose_count = counts.get('loose',0)
    # Add a row in the Excel file for the number of 'win' values and the total amount
    sheet.append(['Number of wins:', win_count])
    sheet.append(['Number of losses:', loose_count])
    sheet.append(['NET real RETURN :', (merged_dataframe["End net trade amount"].iloc[-1]-merged_dataframe["Begin brut trade amount"].iloc[0])/(merged_dataframe["Begin brut trade amount"].iloc[0])])
    # Apply the styles to the sheet
    for cell in sheet["A1":"Z1000"]:
        cell_obj = cell[0]
        row_num, col_num = cell_obj.row, cell_obj.column_letter
        if row_num > 1 and col_num != 'A':  # Skip the header row and the first column
            style = styled_dataframe.data.loc[row_num-2, col_num-2]
            cell_obj._style = style


    workbook.save(filename)

"""## Define the test strategy function"""

def test_strategy(nbpoints, years, months, day, h, min, declenchement, decalage, nbpointsDTW, varMin, varMax,windows1,montant=1000, fees=0.00001):
  percent = 0
  tableau = []
  sum_trade = 0
  transit =0
  trade_en_cours = False
  trade_long = False
  trade_short = False
  print("day :", day, " months :", months)
  results = pd.DataFrame(columns=['Result', 'Montant', 'Day', 'Months','Hours', 'Min','Buy Price', 'Stop Gain', 'Stop Loss', 'Type','Trade returns','Begin brut trade amount','Begin net trade amount', 'End brut trade amount','End net trade amount', 'Begin Fees', 'End Fees'])
  # For Excel data
  #array1vrai = get_day_close_pricesETH2(years, months, day)
  #array2vrai = get_day_close_pricesBTC2(years, months, day)
  # For Yahoo Finance data
  array1vrai = yfv2("GBPUSD", nbpoints,years,months, day, h, min)
  array2vrai = yfv2("EURUSD",nbpoints,years,months, day, h, min)



  nbpoints1 = nbpoints -30
  for i in range(1,nbpoints1):
    array1 = array1vrai[i-1:20+i-1]
    array2= array2vrai[i-1:20+i-1]

    #Mean reverting
    dfmr = pd.DataFrame()
    dfmr["Adj Close"] = array1
    window=windows1
    dfmr['ma_20'] = dfmr['Adj Close'].rolling(window=window).mean()
    dfmr['std_20']  = dfmr['Adj Close'].rolling(window=window).std()
    dfmr['zscore']  = (dfmr['Adj Close'] - dfmr['ma_20']) / dfmr['std_20']
    n_std=1.25
    dfmr['signal'] = np.where(dfmr['zscore'] < -n_std, 1, np.where(dfmr['zscore'] > n_std,-1, 0))
    #end of Mean reverting

    #We start from the index 30 of array1 because we subtract nbpointsDTW, we choose from 30 and this can vary between 7 and 20

    # Attempt to improve triggering a trade
    array1_max = np.mean(array1)
    array1_min = np.mean(array1)
    pips_changesB = [(array2[i] - array2[i-1]) / 0.0001 for i in range(1, len(array2))]
    pips_changesA = [(array1[i] - array1[i-1]) / 0.0001 for i in range(1, len(array1))]

    # Metric calculation and update
    positive_pips_changes = [change for change in pips_changesB if change > 0]
    positive_pips_changes.sort()
    index = int(len(positive_pips_changes) * declenchement)
    if len(positive_pips_changes) != 0:
      max_value_pc = positive_pips_changes[index]
    else : max_value_pc = 10
    negative_pips_changes = [change for change in pips_changesB if change < 0]
    negative_pips_changes.sort()
    index = int(len(negative_pips_changes) *(1-declenchement))
    if len(negative_pips_changes) != 0:
      min_value_pc = negative_pips_changes[index]
    else : min_value_pc = -10
    if(len(pips_changesA)!=0):
      meanA = sum(pips_changesA) / len(pips_changesA)
    else : meanA =0
    positive_pipsA = [pips for pips in pips_changesA if pips > 0]
    negative_pipsA = [pips for pips in pips_changesA if pips < 0]
    mean_positiveA = sum(positive_pipsA) / len(positive_pipsA) if positive_pipsA else 0
    mean_negativeA = sum(negative_pipsA) / len(negative_pipsA) if negative_pipsA else 0
    MaxPositifA = max(positive_pipsA) if positive_pipsA else mean_positiveA
    MinNegativeA = max(abs(pips) for pips in negative_pipsA) if negative_pipsA  else mean_negativeA
    array1 = array1vrai[20-nbpointsDTW+i-1:20+i-1]
    array2 = array2vrai[20-nbpointsDTW+i:20+i-1]


# Case where we don't have any trade
    if trade_en_cours == False:
      normalized_array1 = normalize(array1)
      normalized_array2 = normalize(array2)
      #plot_graph(array1, array2, normalized_array1, normalized_array2)
      pipsA = (array1[len(array1)-1] - array1[len(array1)-2]) / 0.0001
      pipsB = (array2[len(array2)-1] - array2[len(array2)-2]) / 0.0001

      # Short the pair GBPUSD
      if pipsB < (min_value_pc) :
        # Normalization of our two arrays
        a = normalize(array1)
        b = normalize(array2)
        # Calculations related to DTW
        path, cost_matrix, acc_cost_matrix = cost_matrix_compute(a,b)
        deformed_a = [a[p[0]] for p in path[:]]
        x = [p[1] for p in path] # Correspond to the deformed_b
        y = [p[0] for p in path] # Correspond to the index of arrayB
        #plot_deformed(a,b,deformed_a)
        time = len(array2)-1

        # Choice of metrics to trigger a trade
        exec = meanA
        
        #df is a dataframe which will be completed each time there is a small offset which has been established thanks to the DTW between the GPBUSD and the EURUSD
        #when the dtw says that there is a significant delay this means that it is malfunctioning
        if(x[time]-y[time]<decalage and x[time]-y[time]>-decalage):
          new_index = len(df)
          df.at[new_index, 'time'] = time
          df.at[new_index, '1 min'] = pipsB
          if y[time] < x[time]:  #
              df.at[new_index,'Indication A'] = 'D' # This means that array1 has been decompressed = it is ahead of array2
              df.at[new_index, 'Indication B'] = 'C' # This means that array1 has been compressed = it is behind of array1
              df.at[new_index, 'delay'] = x[time] - y[time] # The number of minutes late
          elif y[time] > x[time]:
              df.at[new_index,'Indication A'] = 'C'# This means that array1 has been compressed = it is behind of array2
              df.at[new_index, 'Indication B'] = 'D' # This means that array2 has been decompressed = it is ahead of array1
              df.at[new_index, 'delay'] = x[time] - y[time] # The number of minutes late
          else:
              df.at[new_index,'Indication A'] = 'N' # Array1 and array2 are at the same speed
              df.at[new_index, 'Indication B'] = 'N'# Array1 and array2 are at the same speed
              df.at[new_index, 'delay'] = x[time] - y[time]
          last_row = df.iloc[-1]

          if(pips_changesA[len(pips_changesA)-1] >= exec) and dfmr.iloc[19]['signal']==-1: 
              if(last_row['Indication A'] == 'C' and last_row['1 min'] < 0): # Array1 is late
                sell_price = array1[len(array1)-1] # I short A
                trade_en_cours = True
                stop_loss = sell_price+((MaxPositifA*varMin)*0.0001)
                stop_gain = sell_price +((-MinNegativeA*varMax)*0.0001)
                sum_trade +=1 # Number of trade
                print("le stop gain est ", stop_gain)
                print("le stop loss est ", stop_loss)
                print("le buy price est ", sell_price)
                trade_short = True
              # Array1 (A) lags behind Array2 (B) and performance of B is positive, I buy A
                print("pips_changesB[len(pips_changesB)-1]",pips_changesB[len(pips_changesB)-1])
                print("array1(len",array1[len(array1)-1])
                
    #Long the pair GBPUSD
      if pipsB > (max_value_pc) :

        # Normalization of our two arrays
        a = normalize(array1)
        b = normalize(array2)

        # Calculations related to DTW
        path, cost_matrix, acc_cost_matrix = cost_matrix_compute(a,b)
        deformed_a = [a[p[0]] for p in path[:]]
        x = [p[1] for p in path]
        y = [p[0] for p in path]
        #plot_deformed(a,b,deformed_a)
        time = len(array2)-1

        # Choice of metrics to trigger a trade
        exec= meanA


#df is a dataframe which will be completed each time there is a small offset which has been established thanks to the DTW between the GPBUSD and the EURUSD
#when the dtw says that there is a significant delay this means that it is malfunctioning
        if(x[time]-y[time]<decalage and x[time]-y[time]>-decalage):
          new_index = len(df)
          df.at[new_index, 'time'] = time
          df.at[new_index, '1 min'] = pipsB

          # Check if y < x
          if y[time] < x[time]:
            df.at[new_index,'Indication A'] = 'D' # This means that array1 has been decompressed = it is ahead of array2
            df.at[new_index, 'Indication B'] = 'C' # This means that array1 has been compressed = it is behind of array1
            df.at[new_index, 'delay'] = x[time] - y[time] # The number of minutes late

              # Vérifier si y > x
          elif y[time] > x[time]:
            df.at[new_index,'Indication A'] = 'C' # This means that array1 has been compressed = it is behind of array2
            df.at[new_index, 'Indication B'] = 'D' # This means that array2 has been decompressed = it is ahead of array1
            df.at[new_index, 'delay'] = x[time] - y[time] # The number of minutes late
          else:
            df.at[new_index,'Indication A'] = 'N' # Array1 and array2 are at the same speed
            df.at[new_index, 'Indication B'] = 'N' # Array1 and array2 are at the same speed
            df.at[new_index, 'delay'] = x[time] - y[time]
          last_row = df.iloc[-1]  

          # Buying GPBUSD
          if(pips_changesA[len(pips_changesA)-1] <= exec) and dfmr.iloc[19]['signal']==1:
            if(last_row['Indication A'] == 'C' and last_row['1 min'] > 0):
              percentofvariationB = array1[len(array2)-1] - array1[len(array2)-2] / ( array1[len(array2)-2])
              buy_price = array1[len(array1)-1]
              trade_en_cours = True
              stop_loss = buy_price +((varMin*(-MinNegativeA))*0.0001) 
              stop_gain = buy_price +((varMax*MaxPositifA)*0.0001)
              sum_trade +=1
              print("le stop gain est ", stop_gain)
              print("le stop loss est ", stop_loss)
              print("le buy price est ", buy_price)
              trade_long = True
            #Array1 (A) lagging behind Array2 (B) and performance of B positive, I buy A

    else : # Case where there is already a trade
      tableau.append(array1[len(array1)-1])

      if array1[len(array1)-1] >= stop_gain and trade_long == True: # Stop gain exceeded in case of long = WIN
        gain = stop_gain - buy_price
        trade_return =gain/buy_price #Calculate the return on the trade (in %)
        brut_init_amount = montant #Initial brut investment of the trade
        init_fees = brut_init_amount * fees #Initial amount of fees
        net_init_amount = brut_init_amount - init_fees #Initial net investment of the trade
        brut_end_amount = net_init_amount*(1 + trade_return)
        end_fees = brut_end_amount *fees
        net_end_amount = brut_end_amount - end_fees
        montant = net_end_amount
        print("END OF TRADE, YOU HAVE WON", gain)
        trade_en_cours = False
        trade_long = False
        results = results._append({'Result': 'win', 'Montant': gain, 'Buy Price': buy_price, 'Stop Gain': stop_gain, 'Stop Loss': stop_loss, 'Type': 'Long', 'Day': day, 'Months' : months, 'Hours': ((20+i-2)// 60), 'Min': (20+i-2)% 60 ,'Trade returns' : trade_return,'Begin brut trade amount': brut_init_amount,'Begin net trade amount': net_init_amount, 'End brut trade amount': brut_end_amount,'End net trade amount':net_end_amount, 'Begin Fees':init_fees, 'End Fees':end_fees}, ignore_index=True)

      if array1[len(array1)-1] <= stop_loss and trade_long == True :# Stop loss exceeded in case of long = LOSS
        loss = stop_loss - buy_price
        trade_return =loss/buy_price #Calculate the return on the trade (in %)
        brut_init_amount = montant #Initial brut investment of the trade
        init_fees = brut_init_amount * fees #Initial amount of fees
        net_init_amount = brut_init_amount - init_fees #Initial net investment of the trade
        brut_end_amount = net_init_amount*(1 + trade_return)
        end_fees = brut_end_amount *fees
        net_end_amount = brut_end_amount - end_fees
        montant = net_end_amount
        print("END OF TRADE, YOU HAVE LOST", loss)
        trade_en_cours = False
        trade_long = False
        results = results._append({'Result': 'loose', 'Montant': loss, 'Buy Price': buy_price, 'Stop Gain': stop_gain, 'Stop Loss': stop_loss, 'Type': 'Long', 'Day': day, 'Months' : months, 'Hours': ((min+20+i-2)// 60), 'Min': (20+i-2)% 60, 'Trade returns' : trade_return,'Begin brut trade amount': brut_init_amount,'Begin net trade amount': net_init_amount, 'End brut trade amount': brut_end_amount,'End net trade amount':net_end_amount, 'Begin Fees':init_fees, 'End Fees':end_fees}, ignore_index=True)

      if array1[len(array1)-1] <= stop_gain and trade_short == True: # Stop gain exceeded in case of short = WIN
        gain = sell_price - stop_gain
        trade_return =gain/sell_price #Calculate the return on the trade (in %)
        brut_init_amount = montant #Initial brut investment of the trade
        init_fees = brut_init_amount * fees #Initial amount of fees
        net_init_amount = brut_init_amount - init_fees #Initial net investment of the trade
        brut_end_amount = net_init_amount*(1 + trade_return)
        end_fees = brut_end_amount *fees
        net_end_amount = brut_end_amount - end_fees
        montant = net_end_amount
        print("END OF TRADE, YOU HAVE WON", gain)
        trade_en_cours = False
        trade_short = False
        results = results._append({'Result': 'win', 'Montant': gain, 'Buy Price': sell_price, 'Stop Gain': stop_gain, 'Stop Loss': stop_loss, 'Type': 'Short','Day': day, 'Months' : months, 'Hours': ((min+20+i-2)// 60), 'Min': (20+i-2)% 60, 'Trade returns' : trade_return,'Begin brut trade amount': brut_init_amount,'Begin net trade amount': net_init_amount, 'End brut trade amount': brut_end_amount,'End net trade amount':net_end_amount, 'Begin Fees':init_fees, 'End Fees':end_fees}, ignore_index=True)

      if array1[len(array1)-1] >= stop_loss and trade_short == True : # Stop loss exceeded in case of short = LOSS
        loss = sell_price - stop_loss
        trade_return =loss/sell_price #Calculate the return on the trade (in %)
        brut_init_amount = montant #Initial brut investment of the trade
        init_fees = brut_init_amount * fees #Initial amount of fees
        net_init_amount = brut_init_amount - init_fees #Initial net investment of the trade
        brut_end_amount = net_init_amount*(1 + trade_return)
        end_fees = brut_end_amount *fees
        net_end_amount = brut_end_amount - end_fees
        montant = net_end_amount
        print("END OF TRADE, YOU HAVE LOST", loss)
        trade_en_cours = False
        trade_short = False
        results = results._append({'Result': 'loose', 'Montant': loss, 'Buy Price': sell_price, 'Stop Gain': stop_gain, 'Stop Loss': stop_loss, 'Type': 'Short','Day': day, 'Months' : months, 'Hours': ((min+20+i-2)// 60), 'Min': (20+i-2)% 60, 'Trade returns' : trade_return,'Begin brut trade amount': brut_init_amount,'Begin net trade amount': net_init_amount, 'End brut trade amount': brut_end_amount,'End net trade amount':net_end_amount, 'Begin Fees':init_fees, 'End Fees':end_fees}, ignore_index=True)
  return results, montant

"""## Testing our strategy"""

#Show on google colab with colors
def color_row(row):
    if row['Result'] == 'win':
        return ['background-color: green'] * len(row)
    elif row['Result'] == 'loose':
        return ['background-color: red'] * len(row)
    else:
        return [''] * len(row)
    
def test(declenchement,decalage, nbpointsDTW, varMin, varMax,windows1):
    merged_dataframe = pd.DataFrame()
    montant = 1000
    #warning : we can load just last months data with frequence of 1 minute in yahoo finance
    for i in range(14, 29, 1): #for example february
        try:
            test_strategy1, montant= test_strategy(1300, 2024, 2,i , 1, 0, declenchement, decalage, nbpointsDTW, varMin, varMax,windows1,montant)
            merged_dataframe = pd.concat([merged_dataframe, test_strategy1])
        except Exception as e:
            print(f"Error for the day {i} of months 1 :", e)
            continue
    for i in range(1, 13, 1): #for mars
        try:
            test_strategy1, montant= test_strategy(1300, 2024, 3,i , 1, 0, declenchement, decalage, nbpointsDTW, varMin, varMax,windows1,montant)
            merged_dataframe = pd.concat([merged_dataframe, test_strategy1])
        except Exception as e:
            print(f"Error for the day {i} of months 1 :", e)
            continue
    merged_dataframe = merged_dataframe.reset_index().rename(columns={'index': 'Strategy'})
    merged_dataframe = merged_dataframe.drop("Strategy", axis=1)
    styled_dataframe = merged_dataframe.style.apply(color_row, axis=1)
    clear_output(wait=True)
    print(styled_dataframe)
    save_styled_dataframe_to_excel(styled_dataframe,merged_dataframe, "file_jouroptimal14MARS.xlsx", declenchement, decalage,nbpointsDTW,varMin, varMax,windows1)
    merged_dataframe.loc[0, 'End net trade amount'] = 1000
    plot_soussous = merged_dataframe['End net trade amount']
    print(print(merged_dataframe['End net trade amount']))
    plt.plot(plot_soussous)

# Round the y-axis values to 10^-2
    yticks = np.round(plt.yticks()[0], 2)
    plt.yticks(yticks)
#plot the variation of our portfolio
    plt.show()


'''
#Find optimal parameters
for declenchement in np.arange(0.7, 0.71,0.05):
    for decalage in range(3,4,1):
            for nbpointsDTW in range(7,16,3):
                for varMin in np.arange(0.2, 0.5, 0.1):
                    for varMax in np.arange(0.6, 0.9, 0.1):
                        for windows1 in range(10, 21, 5):
                            test(declenchement,decalage,nbpointsDTW,varMin,varMax,windows1)

for declenchement in np.arange(0.75, 0.9,0.05):
    for decalage in range(2,4,1):
            for nbpointsDTW in range(7,16,3):
                for varMin in np.arange(0.2, 0.5, 0.1):
                    for varMax in np.arange(0.6, 0.9, 0.1):
                        for windows1 in range(10, 21, 5):
                            test(declenchement,decalage,nbpointsDTW,varMin,varMax,windows1)
'''

#Single test
test(0.75,3,10,0.4,0.8,20)
test(0.65,3,10,0.4,0.8,20)
test(0.75,3,10,0.4,0.9,15)
test(0.6,3,10,0.4,0.9,10)
