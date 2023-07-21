import plotly.graph_objects as go  # Library for creating interactive plots
import numpy as np
import os
def plot(df,x_feature_name,y_feature_name,title,preview=False):
    """
    This function takes two dataframes as input and plots the number of calls per day and per week.

    Args:
    daily_df (pandas.DataFrame): A dataframe containing daily call data.
    weekly_df (pandas.DataFrame): A dataframe containing weekly call data.

    Returns:
    None
    """

    # A new instance of the go.Figure() class from the plotly.graph_objects library is created. This will be used to create the plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[x_feature_name],
            y=df[y_feature_name],
            name=y_feature_name,
            mode='lines+markers'
        ))

 

    # Update xaxis properties
    # The x-axis and y-axis titles are updated using the update_xaxes() and update_yaxes() methods of the figure object.
    fig.update_xaxes(title_text='Date')

    # Update yaxis properties
    fig.update_yaxes(title_text=y_feature_name)

    # Update title and height
    # The layout of the figure is updated using the update_layout() method. The title, height, and width of the plot are set.
    fig.update_layout(
        title=f'{title}',
        height=500,
        width=1200
    )

    # Show the plot
    # The plot is displayed using the show() method of the figure object.
    if preview == True:
        return fig 
    else:
        return fig
import plotly.graph_objects as go

def plot_with_difference_curve(predicted, actual,count,msg):
    # Create the trace for the first array
    trace1 = go.Scatter(
        x=list(range(len(predicted))),
        y=predicted,
        mode='lines+markers',
        name='Predicted'
    )

    # Create the trace for the second array
    trace2 = go.Scatter(
        x=list(range(len(actual))),
        y=actual,
        mode='lines+markers',
        name='Actual'
    )

    # # Calculate the difference between the arrays
    # difference = [np.abs(a - b) for a, b in zip(predicted, actual)]

    # # Create the trace for the difference curve                                                 
    # trace_diff = go.Scatter(
    #     x=list(range(len(difference))),
    #     y=difference,
    #     mode='lines+markers',
    #     name='Difference'
    # )

    # Create the layout
    layout = go.Layout(
        title=f'Error Metics : {msg}',
        xaxis=dict(title='Index'),
        yaxis=dict(title='Value'),
        height=600,
        width=1200
    )

    # Create the figure and add the traces
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    # fig.show()
    # Show the figure
    fig.write_image(f'Plots_lstm/{count}.png')