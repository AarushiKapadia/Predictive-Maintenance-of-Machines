#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller, kpss 
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC
import catboost as cb
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostRegressor, CatBoostClassifier
from catboost import CatBoostClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)

import streamlit as st
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def page1():
    st.title(":blue[DATA COLLECTION AND PRE-PROCESSING]")
    # st.write("Welcome to Page 1")
    def main():
        df = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_telemetry.csv")
        df1 = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_machines.csv")
        df2 = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_failures.csv")
        df3 = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_errors.csv")
        df4 = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_maint.csv")
        
        # Display the Telemetry Data
        st.subheader("Telemetry Data")
        st.dataframe(df)
        # st.divider()
        # Display the Machines Data
        st.subheader("Machines Data")
        st.dataframe(df1)

        st.subheader("Failure Data")
        st.dataframe(df2)

        st.subheader("Errors Data")
        st.dataframe(df3)
        
        st.subheader("Maintenance Data")
        st.dataframe(df4)
        # st.divider()

        df['datetime'] = pd.to_datetime(df['datetime'])

        telemetry_daily = df.groupby(['machineID', pd.Grouper(key='datetime', freq='D')]).sum().reset_index()

        telemetry_daily['pressure'] = telemetry_daily['pressure'] / 24
        telemetry_daily['volt'] = telemetry_daily['volt'] / 24
        telemetry_daily['vibration'] = telemetry_daily['vibration'] / 24
        telemetry_daily['rotate'] = telemetry_daily['rotate'] / 24

        telemetry_daily = telemetry_daily.dropna()

        # Display the shape of the DataFrame
        # st.write("Shape of Telemetry df after resampling:", telemetry_daily.shape)

        # Display the head of the DataFrame
        # st.write("Head of telemetry_daily DataFrame:")
        st.subheader("Telemetry Daily Data")
        st.dataframe(telemetry_daily)

        # Preprocessing
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[df['datetime'].dt.year != 2016]
        df = df.sort_values(by='datetime')
        # st.header("PdM Telemetry Data Exploration")


        # Preprocessing
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[df['datetime'].dt.year != 2016]
        df = df.sort_values(by='datetime')


################################################
        # Load the dataset

        tele = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_telemetry.csv")

        # Display the dataset in the app

        # st.title("Outlier Detection and Removal")
        st.header("Outlier Detection and Replacement")

        st.write("Dataset")

        st.dataframe(tele.head())
        # Analyze the 'vibration' column

        st.subheader("Boxplot Analysis")

        df_pressure = tele['pressure']

        median_pressure = df_pressure.median()

        # st.write("Median:", median_pressure)
        # Display the boxplot for 'vibration'

        # st.subheader("Boxplot")

        fig_pressure = sns.boxplot(df_pressure)

        st.pyplot()

        # Calculate the boundaries for outliers detection in 'vibration'

        Q1_vibration = tele['pressure'].quantile(0.25)

        Q3_vibration = tele['pressure'].quantile(0.75)

        IQR_vibration = Q3_vibration - Q1_vibration

        Lower_Fence_pressure = Q1_vibration - (1.5 * IQR_vibration)

        Upper_Fence_pressure = Q3_vibration + (1.5 * IQR_vibration)

        # Update the 'vibration' column based on the boundaries

        tele["pressure"] = np.where(tele["pressure"] >= Upper_Fence_pressure, tele['pressure'].median(), tele['pressure'])
        tele["pressure"] = np.where(tele["pressure"] <= Lower_Fence_pressure, tele['pressure'].median(), tele['pressure'])

        df_pressure = tele['pressure']
        median_pressure = df_pressure.median()
        
        st.subheader(" Boxplot after outlier removal")
        fig_pressure = sns.boxplot(df_pressure)
        st.pyplot()
        # Display the boundaries for outliers detection in 'vibration'
        st.subheader("Pressure Outliers Boundaries")
        st.write("Min:", Lower_Fence_pressure)
        st.write("Max:", Upper_Fence_pressure)
        st.write("Median:", median_pressure)



########
 
        st.divider()
        st.title(" :blue[Trend over the year]")
        st.subheader("Hourly Variation of Pressure over Time")
        plt.figure(figsize=(20, 10))
        plt.plot(df['datetime'], df['pressure'], color='midnightblue')
        plt.xlabel("Datetime")
        plt.ylabel("Pressure")
        plt.title("Pressure Variation over Time")
        st.pyplot()

        max_pressure = df['pressure'].max()
        min_pressure = df['pressure'].min()

        max_pressure_date = df.loc[df['pressure'].idxmax(), 'datetime']
        min_pressure_date = df.loc[df['pressure'].idxmin(), 'datetime']

        st.write("Maximum Pressure:", max_pressure)
        st.write("Date of Maximum Pressure:", max_pressure_date)
        st.write("Minimum Pressure:", min_pressure)
        st.write("Date of Minimum Pressure:", min_pressure_date)


        st.subheader("Statistical Summary of Telemetry Attributes")

 
####### TABLE
        data = {
            'Attribute': ['Pressure', 'Vibration','Voltage','Rotation'],
            'Max': ['185.951997730866','76.7910723016723','255.124717259791','695.020984403396'],
            'Date (Max)':['2015-04-04 21:00:00','2015-04-06 04:00:00','2015-11-16 07:00:00','2015-10-27 22:00:00'],
            'Min': ['51.2371057734253','14.877053998383','97.333603782359','138.432075304341'],
            'Date (Min)':['2015-09-22 00:00:00','2015-07-04 06:00:00','2015-08-31 04:00:00','2015-09-25 08:00:00']
        }

        df = pd.DataFrame(data)
        st.table(df.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
            'selector': 'th',
            'props': [('text-align', 'center')]
        }]))

        
        telemetry_daily['month'] = telemetry_daily['datetime'].dt.month
        monthly_pressure = telemetry_daily.groupby('month')['pressure'].mean()

        monthly_pressure = telemetry_daily.groupby('month')['pressure'].sum()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        #### MACHINE ID 

        st.subheader("Variation of Telemetry attributes (mean) wrt Machine ID")
        mean_pressure_by_machine = telemetry_daily.groupby('machineID')['pressure', 'rotate', 'vibration', 'volt'].mean()
        st.write(mean_pressure_by_machine) 
        
        
        #line plot for monthly variation of vibration: taking the sum of pressure of each month

        monthly_vibration = telemetry_daily.groupby('month')['pressure'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


        plt.figure(figsize=(20, 10))
        plt.plot(monthly_vibration.index, monthly_vibration.values, color='midnightblue', marker='o')

        plt.title('Monthly Variation of Pressure')
        plt.xlabel('Month')
        plt.ylabel('Vibration')
        plt.xticks(monthly_vibration.index, month_names)
        plt.grid(True)
        st.subheader("Monthly Variation of Pressure")
        st.pyplot()

# ....................
        # the value indicates that there is a positive correlation between the two attributes 
        correlation = telemetry_daily['pressure'].corr(telemetry_daily['vibration'])
        # st.header("Correlation between Pressure and Vibration: ")
        # st.write(correlation)

        telemetry_daily['datetime'] = telemetry_daily['datetime'].astype(str)
        telemetry_daily['datetime'] = pd.to_datetime(telemetry_daily['datetime'])
        telemetry_daily.set_index('datetime', inplace=True)


        # Calculate rolling mean for specified columns
        st.divider()
        st.header("Stationarity and Seasonality of data")
        st.subheader("Rolling Mean")
        rolling_mean = telemetry_daily[['pressure', 'vibration', 'rotate', 'volt']].rolling(window=3).mean()

        # Display the rolling mean DataFrame
        st.write(rolling_mean)



        
        # columns = ['pressure', 'volt', 'rotate', 'vibration']
        # rolling_mean = df[columns].rolling(window=18, min_periods=1).mean()
        # st.write(rolling_mean)

        attribute_columns = ['pressure', 'vibration', 'rotate', 'volt']

        # Calculate the rolling mean for the attribute columns in the weekly data
        rolling_mean = telemetry_daily[attribute_columns].resample('D').mean().rolling(window=3).mean()

        # Create the plot
        plt.figure(figsize=(10, 6))
        for col in attribute_columns:
            plt.plot(rolling_mean.index, rolling_mean[col], label=f'Rolling Mean ({col})')

        # Customize the plot
        plt.xlabel('Week')
        plt.ylabel('Mean Value')
        plt.title('Rolling Mean for Daily Data')
        plt.legend()

        # Show the plot
        # st.pyplot()

        # attribute_columns = ['pressure', 'vibration', 'rotate', 'volt']

        # Calculate the rolling standard deviation for the attribute columns in the daily data
        # st.subheader("Rolling Standard Deviation")
        # rolling_std = df[attribute_columns].resample('D').mean().rolling(window=7).std()

        # # Create the plot
        # plt.figure(figsize=(10, 6))
        # for col in attribute_columns:
        #     plt.plot(rolling_std.index, rolling_std[col], label=f'Rolling Standard Deviation ({col})')

        # # Customize the plot
        # plt.xlabel('Date')
        # plt.ylabel('Standard Deviation')
        # plt.title('Rolling Standard Deviation for Daily Data')
        # plt.legend()

        # # Show the plot
        # st.pyplot()
        
       
        st.subheader("KPSS statistics")
        attribute_columns = ['pressure', 'rotate', 'vibration', 'volt']
        results = []
        for column in attribute_columns:
            data = telemetry_daily[column]
            result = kpss(data)
            kpss_statistic = result[0]
            p_value = result[1]
            results.append({'Attribute': column, 'KPSS Statistic': kpss_statistic, 'p-value': p_value})

        results_df = pd.DataFrame(results)
        st.table(results_df)
        # st.divider()

        # Stepwise fit summary for ARIMA model 


        #stepwise fit summary for each attribute to finalise ARIMA model  
        # attribute_columns = ['pressure', 'rotate', 'vibration', 'volt']

        # for column in attribute_columns:
        #     daily_data = df[column].resample('D').mean()
        #     train_data = daily_data.loc['2015-01-01':'2016-01-01']
        #     model = pm.auto_arima(train_data, seasonal=False, trace=True, stepwise=True)
        #     st.header(f"ARIMA model for '{column}':")
        #     st.write(model.summary())

        st.subheader("Summary of ARIMA model")
        # Create a sample data for the table
        data = {
            'Attribute': ['Pressure', 'Rotation', 'Vibration', 'Voltage'],
            'ARIMA Model': ['SARIMAX(0, 0, 3)', 'SARIMAX(0, 0, 2)', 'SARIMAX(0, 0, 2)', 'SARIMAX(0, 0, 1)']
        }

        df = pd.DataFrame(data)

        # Display the table in a 2-column, 5-rows format
        st.table(df.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
            'selector': 'th',
            'props': [('text-align', 'center')]
        }]))

    if __name__ == "__main__":
        main()

def page2():
    st.title(":blue[MACHINES, ERRORS & FAILURES]")
    
    # st.write("Welcome to Page 1")
    def main():
    

        merged_df = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_failures.csv")
        df1 = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_machines.csv")
        merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])

        merged_df['failure'] = merged_df['failure'].astype(str)
        # print(merged_df['failure'])

        label_encoder = LabelEncoder()
        

        merged_df['failure_encoded'] = label_encoder.fit_transform(merged_df['failure'])+1
        # st.write(merged_df.head())
        
        machfail = pd.merge(df1, merged_df)
        # st.write(machfail.head())

        # st.write(machfail.shape)
        machfail['datetime'] = pd.to_datetime(machfail['datetime'])

        # Calculate the minimum datetime value
        min_datetime = machfail['datetime'].min()

        # Convert datetime to days
        machfail['days'] = ((machfail['datetime'] - min_datetime).dt.days)+2

        # Display the updated DataFrame
        # st.write(machfail)
        st.subheader("Failure trend for Machine")

        machine_id = st.number_input("Enter the machine ID:", value=1, min_value=1)
        machine_df = machfail[machfail['machineID'] == machine_id]

        machine_id_1_df = merged_df[merged_df['machineID'] == machine_id]
        
        

        failure_trends = machine_id_1_df.groupby('failure')['failure_encoded'].size()
        plt.figure(figsize=(10,6))
        failure_trends.plot(kind='bar',color='blue')
        plt.title(f"Frequency of errors for Machine ID - {machine_id}")
        plt.show()
        st.pyplot()

        failure_trend = machine_df.groupby('datetime')['failure_encoded'].sum()
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(failure_trend.index, failure_trend.values, marker='o')
        ax.set_title(f"Failure Trend for Machine ID - {machine_id}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Failure')
        ax.grid(True)   
        st.pyplot(fig)

        machine_id_1_df = machfail[machfail['machineID'] == machine_id]
        machine_id_1_df = machine_id_1_df.drop(['model', 'age', ], axis=1)  # Drop 'model' and 'age' columns
        # st.write(machine_id_1_df)

        machine_id_1_df['Difference'] = machine_id_1_df['days'].diff()
        st.write(machine_id_1_df)
        sum_diff=machine_id_1_df['Difference'].sum()
        # st.write(sum_diff)
        mean_diff=sum_diff/len(machine_id_1_df)
        st.write(f":blue[Average number of days between failures for Machine ID: {machine_id} -]", int(mean_diff))
       

        ##################################
        

        df = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_errors.csv")

        # Sort the dataframe by the 'errorID' column

        df = df.sort_values(by='errorID')
        # Convert the 'datetime' column to datetime format

        df['datetime'] = pd.to_datetime(df['datetime'])
        # Filter out rows where the year is not 2016

        df = df[df['datetime'].dt.year != 2016]

        # Encode the 'errorID' column using label encoding

        label_encoder = preprocessing.LabelEncoder()

        df['errorID_encoded'] = label_encoder.fit_transform(df['errorID']) + 1

        errors = df.groupby('errorID')['errorID_encoded'].size()

        plt.figure(figsize=(10, 6))

        errors.plot(kind='bar', color='blue')
        st.subheader("Frequency of Errors")
        plt.title('Frequency of Errors')
        plt.xlabel('ErrorID')
        plt.ylabel('No. of Errors')
        plt.grid(True)
        st.pyplot(plt)
        st.subheader('Monthly Variation of Errors')
        
        # Load data

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['month'] = df['datetime'].dt.month
        monthly_errors = df.groupby('month')['errorID_encoded'].size()

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',

                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


        # Create the plot

        plt.figure(figsize=(20, 10))
        plt.plot(monthly_errors.index, monthly_errors.values, color='blue', marker='o')
        plt.title('Monthly Variation of Errors')
        plt.xlabel('Month')
        plt.ylabel('Errors')
        plt.xticks(monthly_errors.index, month_names)
        plt.grid(True)
        st.pyplot(plt)

  

    ###########################

        avg_errors = df.groupby('machineID')['errorID_encoded'].size()

        plt.figure(figsize=(20,6))

        plt.title('Variation of Errors by machine ID')

        avg_errors.plot(kind='bar')

        st.pyplot(plt)
        
        # avg_errors = df.groupby('machineID')['errorID_encoded'].size()

        # # Find the top 3 machine IDs with the most errors
        # avg_errors = df.drop('errorID_encoded', axis=1).groupby('machineID').size()
        # top_3_most_errors = avg_errors.nlargest(3)

        # # Find the top 3 machine IDs with the least errors
        # top_3_least_errors = avg_errors.nsmallest(3)

        # # Print the results
        # st.write("Top 3 Machine IDs with Most Errors:")
        # st.write(top_3_most_errors)

        # st.write("Top 3 Machine IDs with Least Errors:")
        # st.write(top_3_least_errors)

        avg_errors = df.groupby('machineID')['errorID_encoded'].size()

        # Find the top 3 machine IDs with the most errors
        avg_errors = df.drop('errorID_encoded', axis=1).groupby('machineID').size()
        top_3_most_errors = avg_errors.nlargest(3)

        # Find the top 3 machine IDs with the least errors
        top_3_least_errors = avg_errors.nsmallest(3)

        # Rename the column headers
        top_3_most_errors = top_3_most_errors.rename_axis('Machine ID').reset_index(name='No of errors')
        top_3_least_errors = top_3_least_errors.rename_axis('Machine ID').reset_index(name='No of errors')

        # Print the results
        st.write("Top 3 Machine IDs with Most Errors:")
        st.write(top_3_most_errors)

        st.write("Top 3 Machine IDs with Least Errors:")
        st.write(top_3_least_errors)


        # st.write("Machines with the Most and Least Error")
        # tab1 = pd.DataFrame({
        #     'Maximum Errors': ['22', '78', '99', '15'],
        #     'Minimum Erros': ['77', '6', '31', '86']
        # })
        # st.table(tab1)

    
        df1 = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_machines.csv")

        df2=pd.merge(df,df1) #machines and failure

        # st.write(df2)

        df3= pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_maint.csv")

        # st.write(df3) #maintenance

        df4= pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_telemetry.csv")

        df4['datetime']=pd.to_datetime(df4['datetime'])

        # st.write(df4)

        mer=pd.merge(df2,df4) #errors machines telemery combined 

        # st.write(mer)

        plt.figure(figsize=(10,8))

        sns.heatmap(mer.corr())

        # st.subheader('Correlation matrix')
        # st.write("After merging the following files, this is the correlation matrix for Telemetry, Errors and Machines dataset")

        # st.pyplot(plt)

#__________________

        fig, ax = plt.subplots()
        plt.bar(mer["age"],mer["pressure"])
        ax.set_ylabel('pressure')
        ax.set_xlabel('Age')
        ax.set_title('Pressure with Age')
        plt.show()
        st.pyplot(plt)
        max_pressure_age = mer['age'][mer['pressure'].idxmax()]
        st.write(f"Age with Maximum Pressure: {max_pressure_age}")
    

        fig, ax = plt.subplots()
        plt.bar(mer["age"],mer["vibration"])
        ax.set_ylabel('Vibration')
        ax.set_xlabel('Age')
        ax.set_title('Vibration with Age')
        plt.show()
        st.pyplot(plt)
        max_pressure_age = mer['age'][mer['vibration'].idxmax()]
        st.write(f"Age with Maximum Vibration: {max_pressure_age}")
    
        fig, ax = plt.subplots()
        plt.bar(mer["age"],mer["rotate"])
        ax.set_ylabel('Rotation')
        ax.set_xlabel('Age')
        ax.set_title('Rotation with Age')
        plt.show()
        st.pyplot(plt)
        max_pressure_age = mer['age'][mer['rotate'].idxmax()]
        st.write(f"Age with Maximum Rotation: {max_pressure_age}")
    
        fig, ax = plt.subplots()
        plt.bar(mer["age"],mer["volt"])
        ax.set_ylabel('Voltage')
        ax.set_xlabel('Age')
        ax.set_title('Voltage with Age')
        plt.show()
        st.pyplot(plt)
        max_pressure_age = mer['age'][mer['volt'].idxmax()]
        st.write(f"Age with Maximum Voltage: {max_pressure_age}")


        # fig, ax = plt.subplots()
        # ax.bar(mer["age"], mer["pressure"], label='Pressure')
        # ax.set_ylabel('Pressure')
        # ax.set_xlabel('Age')
        # ax.set_title('Pressure with Age')

        # ax.plot(mer["age"], mer["pressure"], color='red', label='Line Plot')
        # ax.legend()
        # st.pyplot(plt)

        fig, ax = plt.subplots()
        plt.bar(mer["errorID_encoded"],mer["pressure"])
        ax.set_ylabel('Pressure')
        ax.set_xlabel('ErrorID')
        ax.set_title('Pressure with Error')
        plt.show()
        st.pyplot(plt)

        fig, ax = plt.subplots()
        plt.bar(mer["errorID_encoded"],mer["vibration"])
        ax.set_ylabel('Vibration')
        ax.set_xlabel('ErrorID')
        ax.set_title('Vibration with Error')
        plt.show()
        st.pyplot(plt)

        fig, ax = plt.subplots()
        plt.bar(mer["errorID_encoded"],mer["rotate"])
        ax.set_ylabel('Rotation')
        ax.set_xlabel('ErrorID')
        ax.set_title('Rotation with Error')
        plt.show()
        st.pyplot(plt)

        fig, ax = plt.subplots()
        plt.bar(mer["errorID_encoded"],mer["volt"])
        ax.set_ylabel('Voltage')
        ax.set_xlabel('ErrorID')
        ax.set_title('Voltage with Error')
        plt.show()
        st.pyplot(plt)

        

        mer['date'] = mer['datetime'].dt.date

        # date=df.groupby('datetime')['volt'].sum()

        error=mer.groupby('date')['errorID_encoded'].size()

       

        # axis_position = plt.axes([0.2, 0.1, 0.65, 0.03],)
        # slider_position = Slider(axis_position,'Pos', 0.1, 90.0)
        # def update(val):
        #     pos = slider_position.val
        #     Axis.axis([pos, pos+1, 0, 1])
        #     Plot.canvas.draw_idle()
        # plt.xticks(monthly_pressure.index, month_names)
        # plt.grid(True)




        # plt.figure(figsize=(20, 10))
        # plt.plot(error.index, error.values, color='red')
        # plt.title('Daily Variation of Error')
        # plt.xlabel('Months')
        # plt.ylabel('Errors')
        # # slider_position.on_changed(update)
        # plt.show()
        # st.pyplot(plt)

        plt.figure(figsize=(20, 10))
        plt.plot(error.index, error.values, color='red')
        plt.title('Daily Variation of Error')
        plt.xlabel('Months')
        plt.ylabel('Errors')

        top_3_max_days = error.nlargest(3)
        top_3_min_days = error.nsmallest(3)

        for day in top_3_max_days.index:
            plt.text(day, error[day], f"Max Day: {day}", ha='center', va='bottom')
        for day in top_3_min_days.index:
            plt.text(day, error[day], f"Min Day: {day}", ha='center', va='top')

        st.subheader("Daily Variation of Error")
        plt.show()
        st.pyplot()



    if __name__ == "__main__":
        main()

def page3():
    # st.subheader("Variation of Telemetry Attributes wrt Age of Machine")
    st.title(":blue[Variation of Failure Components wrt Telemetry Attributes]")
    # st.write("Welcome to Page 1")

        ## MACHINES DATASET
    def main():
        df1 = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_machines.csv")
        # st.header("MACHINES DATASET")

        # Read the CSV files
        df = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_telemetry.csv")

        # Convert 'datetime' column to datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Compute daily averages for telemetry data
        telemetry_daily = df.groupby(['machineID', pd.Grouper(key='datetime', freq='D')]).sum().reset_index()

        # Divide telemetry values by 24 to get daily averages
        telemetry_daily['pressure'] = telemetry_daily['pressure'] / 24
        telemetry_daily['volt'] = telemetry_daily['volt'] / 24
        telemetry_daily['vibration'] = telemetry_daily['vibration'] / 24
        telemetry_daily['rotate'] = telemetry_daily['rotate'] / 24

        # Drop rows with missing values
        telemetry_daily = telemetry_daily.dropna()

        # Merge telemetry data with machines data
        merged_df = pd.merge(telemetry_daily, df1, on=['machineID'], how='left')

        # Display the shape and head of the merged DataFrame
        # st.write("Shape of merged DataFrame:", merged_df.shape)
        st.write("Merged DataFrame:")
        st.write(merged_df.head())

        # Mean Pressure Variation by Age
        # mean_pressure_by_age = merged_df.groupby('age')['pressure'].mean()
        # plt.plot(mean_pressure_by_age.index, mean_pressure_by_age.values, marker='o')
        # plt.title('Mean Pressure Variation by Age')
        # plt.xlabel('Age')
        # plt.ylabel('Mean Pressure')
        # st.pyplot(plt)
        
        # # Mean Rotation Variation by Age
        # mean_rotation_by_age = merged_df.groupby('age')['rotate'].mean()
        # plt.plot(mean_rotation_by_age.index, mean_rotation_by_age.values, marker='o')
        # plt.title('Mean rotation Variation by Age')
        # plt.xlabel('Age')
        # plt.ylabel('Mean rotation')
        # st.pyplot()


        # # Calculate mean voltage by age
        # mean_voltage_by_age = merged_df.groupby('age')['rotate'].mean()
        # plt.plot(mean_voltage_by_age.index, mean_voltage_by_age.values, marker='o')
        # plt.title('Mean Rotation Variation by Age')
        # plt.xlabel('Age')
        # plt.ylabel('Mean Rotation')
        # st.pyplot()

        # # Mean voltage Variation by Age
        # mean_vibration_by_age = merged_df.groupby('age')['vibration'].mean()
        # plt.plot(mean_vibration_by_age.index, mean_vibration_by_age.values, marker='o')
        # plt.title('Mean vibration Variation by Age')
        # plt.xlabel('Age')
        # plt.ylabel('Mean vibration')
        # st.pyplot()

        # # Mean Vibration Variation by Age
        # mean_voltage_by_age = merged_df.groupby('age')['volt'].mean()
        # plt.plot(mean_voltage_by_age.index, mean_voltage_by_age.values, marker='o')
        # plt.title('Mean voltage Variation by Age')
        # plt.xlabel('Age')
        # plt.ylabel('Mean voltage')
        # st.pyplot()


        # Calculate the average age per machine
        avg_age = df1.groupby('machineID')['age'].mean()
        plt.figure(figsize=(20, 6))
        avg_age.plot(kind='line', color='pink')
        plt.title('Average Age per Machine')
        plt.xlabel('Machine ID')
        plt.ylabel('Average Age')
        plt.grid(True)
        # st.pyplot()

        

        # Load the telemetry, failures, and additional dataset

        df = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_telemetry.csv")

        df2 = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_failures.csv")

        df1 = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_machines.csv")


        df3 = pd.merge(df, df2, how='outer')
        df4 = pd.merge(df3, df1)
        
        df5=pd.merge(df,df2) #telemetry and failures
        mer = pd.merge(df5,df1) #df5 and machines 

        # st.write(df4)
        # Convert failure column to string type
        df3['failure'] = df3['failure'].astype(str)
        df4['failure'] = df4['failure'].astype(str)

        # Plotting failure only, df1 and df2: 761 rows
        #VARIATION OF FAILURE COMPONENTS WITH TELEMETRY 
        plt.figure(figsize=(10, 6))
        plt.scatter(df5['failure'], df5['pressure'], s=50, alpha=0.5)
        plt.title('Variation of Pressure wrt Failure Components')
        plt.xlabel('Failure Component')
        plt.ylabel('Pressure')
        plt.grid(True)
        st.pyplot()

        plt.figure(figsize=(10, 6))
        plt.scatter(df5['failure'], df5['rotate'], s=50, alpha=0.5)
        plt.title('Variation of Rotation wrt Failure Components')
        plt.xlabel('Failure Component')
        plt.ylabel('Rotation')
        plt.grid(True)
        st.pyplot()

        plt.figure(figsize=(10, 6))
        plt.scatter(df5['failure'], df5['volt'], s=50, alpha=0.5)
        plt.title('Variation of Voltage wrt Failure Components')
        plt.xlabel('Failure Component')
        plt.ylabel('Voltage')
        plt.grid(True)
        st.pyplot()

        plt.figure(figsize=(10, 6))
        plt.scatter(df5['failure'], df5['vibration'], s=50, alpha=0.5)
        plt.title('Variation of Vibration wrt Failure Components')
        plt.xlabel('Failure Component')
        plt.ylabel('Vibration')
        plt.grid(True)
        st.pyplot()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(mer['failure'], mer['age'], s=50, alpha=0.5)
        plt.title('Variation of Failure Components wrt Age')
        plt.xlabel('Failure Component')
        plt.ylabel('Age')
        plt.grid(True)
        st.pyplot()

    if __name__ == "__main__":
        main()


def page4():
    def main():
        # st.title("")
        st.title(":blue[MODEL BUILDING]")
        df = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_telemetry.csv")
        df5= pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_failures.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Group by machineID and daily frequency, and calculate the daily averages
        telemetry_daily = df.groupby(['machineID', pd.Grouper(key='datetime', freq='D')]).sum().reset_index()
        telemetry_daily['pressure'] = telemetry_daily['pressure'] / 24
        telemetry_daily['volt'] = telemetry_daily['volt'] / 24
        telemetry_daily['vibration'] = telemetry_daily['vibration'] / 24
        telemetry_daily['rotate'] = telemetry_daily['rotate'] / 24

        # Drop any rows with missing values
        telemetry_daily = telemetry_daily.dropna()

        # Display shape and head of the DataFrame using Streamlit
        # st.write("Shape of dataset after resampling:", telemetry_daily.shape)
        # st.write("First Few Rows:")
        # st.write(telemetry_daily.head())

        # Replace failure types with binary values
        df5['failure'] = df5['failure'].replace(['comp1', 'comp2', 'comp3', 'comp4'], 1)

        # Convert failure column to string type
        df5['failure'] = df5['failure'].astype(str)

        # # Print the failure column
        # st.write(df5['failure'])

        # Convert datetime column to datetime type and set it as the index
        df5['datetime'] = pd.to_datetime(df5['datetime'])
        df5.set_index('datetime', inplace=True)

        # Group by machineID and daily frequency, and calculate the daily sums
        df5 = df5.groupby(['machineID', pd.Grouper(freq='D')]).sum()
        df5 = df5.reset_index()

        # Normalize the datetime column
        df5['datetime'] = df5['datetime'].dt.normalize()

        # # Display the head of the DataFrame
        # st.write(df5.head())

        merged_df = pd.merge(telemetry_daily, df5, on=['machineID', 'datetime'], how='left')

        st.write("Shape of dataset after merging: ", merged_df.shape)
        # st.write(merged_df.shape)
        # st.write(merged_df.head())
        merged_df['failure'] = merged_df['failure'].astype(str)
        merged_df['failure'] = merged_df['failure'].replace(['comp1', 'comp2', 'comp3', 'comp4',
                                                            'comp2comp4', 'comp2comp3', 'comp1comp2',
                                                            'comp1comp4', 'comp1comp3', 'comp3comp4',
                                                            '1', '11'], 1)
        merged_df['failure'] = merged_df['failure'].replace('nan',0)

        # Display the failure column
        # st.write("Failure column:")
        st.write(merged_df)
        st.divider()

        # st.header("Models")
        st.header("Random Forest Classifier")

        merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])
        merged_df.set_index('datetime', inplace=True)

        # Split the data into features (X) and target (y)
        X = merged_df.drop('failure', axis=1)
        y = merged_df['failure']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')

        # Train the classifier
        rf_classifier.fit(X_train, y_train)

        # Predict on the test set
        y_pred = rf_classifier.predict(X_test)

        # Calculate and st.write accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

        # Calculate and st.write confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # st.write("Confusion Matrix:", cm)

        # Calculate and st.write train accuracy
        y_train_pred = rf_classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        st.write("Train Accuracy:", train_accuracy)

        # Calculate and st.write test accuracy
        y_test_pred = rf_classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        st.write("Test Accuracy:", test_accuracy)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        st.divider()
        # Create a Support Vector Classifier
        classifier = SVC(kernel='rbf', random_state=0)

        # Train the classifier
        classifier.fit(X_train, y_train)

        # Predict on the test set
        y_pred = classifier.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Calculate testing accuracy
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Display the results using Streamlit
        st.header('Support Vector Classifier')
        st.write('Accuracy:', accuracy)
        # st.write('Confusion Matrix:')
        # st.write(cm)
        st.write('Training Accuracy:', train_accuracy)
        st.write('Testing Accuracy:', test_accuracy)

        # LOGISTIC REGRESSION - NEWTON CG 
        st.divider()
        st.header('Logistic Regression - netwon-cg solver')
        lr = LogisticRegression(solver='newton-cg', class_weight='balanced')
        lr.fit(X_train, y_train)

        X = merged_df.drop('failure', axis=1)  
        y = merged_df['failure']

        pred_test = lr.predict(X_test)

        st.write("Accuracy Score " , accuracy_score(y_test,pred_test))
        # st.write("Confusion Matrix:",confusion_matrix(y_test, pred_test))

        #train accuracy 
        y_train_pred = lr.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        st.write("Train Accuracy:", train_accuracy)

        #test accuracy
        y_test_pred = lr.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        st.write("Test Accuracy:", test_accuracy)


        #CATBOOST
        st.divider()
        st.header("CatBoost Classifier")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        class_counts = y_train.value_counts()
        class_weights = class_counts.sum() / (len(class_counts) * class_counts)

        # Create a CatBoost classifier
        catboost_model = cb.CatBoostClassifier(iterations=100,
                                            random_state=42,
                                            eval_metric='Accuracy',
                                            task_type='CPU',
                                            verbose=50,
                                            class_weights=class_weights)

        catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
        y_pred = catboost_model.predict(X_test)
        pred_test = catboost_model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, pred_test)

        # Calculate confusion matrix
        # cm = confusion_matrix(y_test, y_pred)

        # Display the results
        st.write("Accuracy:", accuracy)
        # st.write("Confusion Matrix:")
        # st.write(cm)

        
        # st.write('CatBoost training')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        catboost_model = CatBoostClassifier()
        catboost_model.fit(X_train, y_train)

        y_train_pred = catboost_model.predict(X_train)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        st.write("Train Accuracy:", train_accuracy)

        # st.write('CatBoost testing')
        catboost_model = CatBoostClassifier()
        catboost_model.fit(X_train, y_train)

        y_test_pred = catboost_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        st.write("Test Accuracy:", test_accuracy)
    if __name__ == "__main__":
        main()

def page5():
    def main():

        # merged_df.set_index('datetime', inplace=True)

        st.title(":blue[FAILURE PREDICTION]")
        df = pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_telemetry.csv")
        df5= pd.read_csv("C:/Users/kapad/Downloads/MAPdM/PdM_failures.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Group by machineID and daily frequency, and calculate the daily averages
        telemetry_daily = df.groupby(['machineID', pd.Grouper(key='datetime', freq='D')]).sum().reset_index()
        telemetry_daily['pressure'] = telemetry_daily['pressure'] / 24
        telemetry_daily['volt'] = telemetry_daily['volt'] / 24
        telemetry_daily['vibration'] = telemetry_daily['vibration'] / 24
        telemetry_daily['rotate'] = telemetry_daily['rotate'] / 24

        # Drop any rows with missing values
        telemetry_daily = telemetry_daily.dropna()

        # Display shape and head of the DataFrame using Streamlit
        # st.write("Shape of dataset after resampling:", telemetry_daily.shape)
        # st.write("First Few Rows:")
        # st.write(telemetry_daily.head())

        # Replace failure types with binary values
        df5['failure'] = df5['failure'].replace(['comp1', 'comp2', 'comp3', 'comp4'], 1)

        # Convert failure column to string type
        df5['failure'] = df5['failure'].astype(str)

        # # Print the failure column
        # st.write(df5['failure'])

        # Convert datetime column to datetime type and set it as the index
        df5['datetime'] = pd.to_datetime(df5['datetime'])
        df5.set_index('datetime', inplace=True)

        # Group by machineID and daily frequency, and calculate the daily sums
        df5 = df5.groupby(['machineID', pd.Grouper(freq='D')]).sum()
        df5 = df5.reset_index()

        # Normalize the datetime column
        df5['datetime'] = df5['datetime'].dt.normalize()

        # # Display the head of the DataFrame
        # st.write(df5.head())

        merged_df = pd.merge(telemetry_daily, df5, on=['machineID', 'datetime'], how='left')

        # st.write("Shape of dataset after merging: ", merged_df.shape)
        # st.write(merged_df.shape)
        # st.write(merged_df.head())
        merged_df['failure'] = merged_df['failure'].astype(str)
        merged_df['failure'] = merged_df['failure'].replace(['comp1', 'comp2', 'comp3', 'comp4',
                                                            'comp2comp4', 'comp2comp3', 'comp1comp2',
                                                            'comp1comp4', 'comp1comp3', 'comp3comp4',
                                                            '1', '11'], 1)
        merged_df['failure'] = merged_df['failure'].replace('nan',0)

        # Display the failure column
        # st.write("Failure column:")
        # st.write(merged_df)
        # st.divider()


        X = merged_df[['pressure', 'vibration', 'rotate', 'volt']]
        y = merged_df['failure']

        model = CatBoostClassifier()
        model.fit(X, y)

        failed_records = merged_df[merged_df['failure'] == 1]

        failed_pressure = failed_records['pressure'].values
        failed_vibration = failed_records['vibration'].values
        failed_rotation = failed_records['rotate'].values
        failed_voltage = failed_records['volt'].values

        # st.write("Values of attributes for failed machines:")
        # for i in range(len(failed_pressure)):
        #     st.write(f"Record {i+1}:")
        #     st.write(f"Pressure: {failed_pressure[i]}")
        #     st.write(f"Vibration: {failed_vibration[i]}")
        #     st.write(f"Rotation: {failed_rotation[i]}")
        #     st.write(f"Voltage: {failed_voltage[i]}")
        #     st.write("----------------------")

        no_failure_df = merged_df[merged_df['failure'] == 0]

        pressure_min = no_failure_df['pressure'].min()
        pressure_max = no_failure_df['pressure'].max()

        volt_min = no_failure_df['volt'].min()
        volt_max = no_failure_df['volt'].max()

        rotate_min = no_failure_df['rotate'].min()
        rotate_max = no_failure_df['rotate'].max()

        vibration_min = no_failure_df['vibration'].min()
        vibration_max = no_failure_df['vibration'].max()

        # st.write("Range of Non-Failure Attributes:")
        # st.write(f"Pressure: {pressure_min} - {pressure_max}")
        # st.write(f"Voltage: {volt_min} - {volt_max}")
        # st.write(f"Rotation: {rotate_min} - {rotate_max}")
        # st.write(f"Vibration: {vibration_min} - {vibration_max}")

        pressure = st.number_input("Enter the pressure value:")
        volt = st.number_input("Enter the voltage value:")
        rotation = st.number_input("Enter the rotation value:")
        vibration = st.number_input("Enter the vibration value:")
        submit_button = st.button("Submit")
        if submit_button:
            if (

                ((failed_records['pressure'] == pressure) & (failed_records['vibration'] == vibration) &

                (failed_records['rotate'] == rotation) & (failed_records['volt'] == volt)).any() or

                pressure < pressure_min or pressure > pressure_max or

                vibration < vibration_min or vibration > vibration_max or

                rotation < rotate_min or rotation > rotate_max or

                volt < volt_min or volt > volt_max

            ):

                st.subheader("The machine is predicted to fail.")

            else:

                st.subheader("The machine is predicted to not fail.")

    if __name__ == "__main__":
        main()
            


# Create a dictionary to map page names to their corresponding functions
pages = {
    "Telemetry": page1,
    "Failure + Errors": page2,
    "Machines, Failure + Telemetry": page3, 
    "Model Building": page4,
    "Failure prediction":page5
}

# Create a sidebar or navigation menu
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Display the selected page
pages[selection]()


# if __name__ == "__main__":
#     main()