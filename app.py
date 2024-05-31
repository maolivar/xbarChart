import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import base64


# read file
df = pd.read_csv('df_selected2.csv')
df.head()

# Functions for grouping and graphs
def gen_grouped_data(df, groupvar, filter_hour, filter_dow):
    # Filter the data based on the hour_block and day_of_week
    df_filtered = df[df['hour_block'].isin(filter_hour) & df['day_of_week'].isin(filter_dow)]
    # Group the data by 'driver_id' and calculate the mean and standard deviation of 'driver_rating'
    df_grouped = df_filtered.groupby(groupvar)['driver_rating'].agg(['mean', 'std', 'count'])
    overall_mean = df_filtered['driver_rating'].mean()
    df_grouped['center'] = overall_mean
    overall_std = df_filtered['driver_rating'].std()
    df_grouped['overall_std']=overall_std

    # Reset the index of df_grouped so 'driver_id' becomes a column again
    df_grouped.reset_index(inplace=True)

    return df_grouped


def gen_control_chart(df_grouped, groupvar):
    # Convert driver_id to a categorical data type
    df_grouped[groupvar] = df_grouped[groupvar].astype(str)

    # Calculate the UCL and LCL
    df_grouped['ucl'] = df_grouped['center'] + 3 * df_grouped['overall_std'] / np.sqrt(df_grouped['count'])
    df_grouped['lcl'] = df_grouped['center'] - 3 * df_grouped['overall_std'] / np.sqrt(df_grouped['count'])

    # Plot the control chart
    fig = px.scatter(df_grouped, x=groupvar, y='mean', hover_data=[groupvar,'count'])
    fig.update_xaxes(type='category')

    # Add lines for the upper control limit, lower control limit, and center
    fig.add_trace(go.Scatter(x=df_grouped[groupvar], y=df_grouped['ucl'], mode='lines', name='ucl', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df_grouped[groupvar], y=df_grouped['lcl'], mode='lines', name='lcl', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df_grouped[groupvar], y=df_grouped['center'], mode='lines', name='center', line=dict(color='red', dash='dash')))
    

    return fig, df_grouped


# test first
# list_filter_hour = df['hour_block'].unique().tolist()
# list_filter_dow = df['day_of_week'].unique().tolist()
# list_drivers = df['driver_id'].unique().tolist()

# filter_hour = list_filter_hour
# filter_dow = list_filter_dow
# groupvar = 'driver_id'

# df_grouped = gen_grouped_data(df, groupvar, filter_hour, filter_dow)

# print(df_grouped.head())
# fig, df_graph = gen_control_chart(df_grouped, groupvar)
# fig.show()



# --------- Streamlit code ------------

# Set the title of the app
st.title('Analyzing driver ratings in a ride-hailing platform')

# Add a short description
st.markdown("""
Data has been collected on a sample of rides in a ride-hailing platform in the city of Austin, TX. The data describes information for each ride, including an anonymized driver id and the ratings that the customer provided to the rider.

The following dashboard can be used to build control charts to analyze the driver ratings and identify whether there are special causes that affect their performance.
""")


# Convert 'driver_id' to string
df['driver_id'] = df['driver_id'].astype(str)


# Create a dropdown for groupvar
groupvar = st.selectbox('Group by:', ['driver_id', 'hour_block', 'day_of_week'])

# Create a multiselect widget for filter_hour
filter_hour = st.multiselect('Selected Hour Block:', df['hour_block'].unique())

# Create a multiselect widget for filter_dow
filter_dow = st.multiselect('Selected Day of Week:', df['day_of_week'].unique())

# Create a slider for mincount
mincounter = st.slider('Minimum sample size:', min_value=0, max_value=50, value=10, step=1)

# Add a button for running the functions
run_button = st.button('Generate Xbar Chart')

# Only run the following code when the button is clicked
if run_button:
    # Run your functions here
    df_grouped = gen_grouped_data(df, groupvar, filter_hour, filter_dow)

    # Convert 'driver_id' to integer, sort the values, and convert back to string
    if groupvar == 'driver_id':
        df_grouped['driver_id'] = df_grouped['driver_id'].astype(int)
        df_grouped = df_grouped.sort_values('driver_id')
        df_grouped['driver_id'] = df_grouped['driver_id'].astype(str)

    df_grouped = df_grouped[df_grouped['count']>=mincounter].copy()
    #print(df_grouped.head())
    fig, df_graph = gen_control_chart(df_grouped, groupvar)

    # Display the plot
    st.plotly_chart(fig)

    # Convert the DataFrame to CSV and generate a download link
    csv = df_graph.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

    st.markdown(href, unsafe_allow_html=True)

