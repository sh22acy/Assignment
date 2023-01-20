import wbdata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# import world bank data
dataset_location = 'world_bank_data.csv'
def return_correct_format(dataset_location):
    data = pd.read_csv(dataset_location)
    data = data.set_index('Country Name')
    data = data.drop(['Country Code' , 'Indicator Name' , 'Indicator Code'] , axis=1)
    data = data.T
    return data

# directly import world bank data

# 'BR' for Brazil and 'US' for United States of America
countries = ['BR','US']


# downloading the data
dataset_br = wbdata.get_dataframe({"EG.USE.ELEC.KH.PC" : "Electric power consumption (kWh per capita)"} , 
                              country = countries[0] , 
                              convert_date = False)

dataset_us = wbdata.get_dataframe({"EG.USE.ELEC.KH.PC" : "Electric power consumption (kWh per capita)"} , 
                              country = countries[1] , 
                              convert_date = False)

# drop empty entries
dataset_br.dropna(inplace = True)
dataset_us.dropna(inplace = True)

# comparing the electricity consumption of Brazil & USA using barplot
plt.bar(
        dataset_br.index , 
        dataset_br['Electric power consumption (kWh per capita)'] , 
        color = 'yellow'
       )
plt.ylabel('Years')
plt.xlabel('Electric power consumption')
plt.title("Electric power consumption (kWh per capita) of Brazil")
plt.tick_params(axis = 'x' , labelsize = 6)
plt.show()

plt.clf()

plt.bar(
        dataset_us.index , 
        dataset_us['Electric power consumption (kWh per capita)'] , 
        color = 'red'
       )
plt.ylabel('Years')
plt.xlabel('Electric power consumption')
plt.title("Electric power consumption (kWh per capita) of USA")
plt.tick_params(axis = 'x' , labelsize = 5)
plt.show()

plt.clf()

# normalize the dataframe using the StandardScaler technique
normalized_dataset = StandardScaler().fit_transform(dataset_us)
print(normalized_dataset)

# DBSCAN Clustering
dbscan = DBSCAN(eps = 0.3 , min_samples = 2)
dbscan.fit(normalized_dataset)
labels = dbscan.labels_

dataset_us['Clusters'] = labels

# plot the data
plt.scatter(
            dataset_us.index ,
            dataset_us['Electric power consumption (kWh per capita)'] ,
            c = dataset_us['Clusters'] , 
            cmap = 'rainbow'
           ) 

plt.xlabel('Years')
plt.ylabel('Electric power consumption (kWh per capita))')
plt.title('Visualisation Of Clusters')
plt.tick_params(
                    axis = 'x' , 
                    labelsize = 5
                )
plt.show()

# define the function to be fitted
def model_function(x , a , b):
    return a * np.exp(b * x)

dataset_us['Year'] = dataset_us.index
x_data = np.linspace(0 , 5 , 100)
y_data = model_function(x_data , 2.5 , 1.3)

# applying curve_fit
popt, pcov = curve_fit(
                        model_function , 
                        dataset_us.index , 
                        dataset_us['Electric power consumption (kWh per capita)']
                    )


# get the parameter estimates
a_est, b_est = popt

# predictions
x_pred = np.array([dataset_us.index[-1] + 10])
y_pred = model_function(x_pred , *popt)

# Define the error function
def err_ranges(popt , pcov , x_pred , n_sigma = 1):
    y_fit = model_function(x_pred , *popt)
    s_sq = (dataset_br['Electric power consumption (kWh per capita)'] - y_fit) ** 2 / (len(dataset_us['Electric power consumption (kWh per capita)']) - len(popt))
    ci = n_sigma * np.sqrt(np.diag(pcov) * s_sq)
    return y_fit - ci , y_fit + ci

lower, upper = err_ranges(popt, pcov, x_pred)

plt.scatter(
            dataset_us.index , 
            dataset_us['Electric power consumption (kWh per capita)'] , 
            c = 'blue' , 
            label = 'Original data'
            )
plt.plot(
        x_pred , 
        y_pred , 
        c = 'red' , 
        label = 'Predicted values'
        )
plt.fill_between(
                    x_pred , 
                    lower , 
                    upper , 
                    color = 'gray' , 
                    alpha = 0.2 , 
                    label = 'Confidence range'
                )

plt.xlabel('Years')
plt.ylabel('Electric power consumption (kWh per capita)')
plt.legend()
plt.show()

