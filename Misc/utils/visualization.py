# The code defines a function named plot that takes four parameters: df, x_feature_name,
# y_feature_name, and title. The function is designed to plot data from a dataframe
import plotly.graph_objects as go  # Library for creating interactive plots

def plot(df,x_feature_name,y_feature_name,title):
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
    # Add a trace for daily calls
    # A trace is added to the figure using the go.Scatter() class from plotly.graph_objects. 
    # It specifies the x and y data for the plot, assigns a name to the trace, 
    # and sets the mode to display lines and markers.
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
    fig.show()

    # Write the plot to an HTML file
    # fig.write_html(f'Visualization/btc.html')
# Summary:
# The code defines a function plot() that takes in a dataframe (df), x-axis feature name (x_feature_name),
# y-axis feature name (y_feature_name), and a title. Inside the function, a plot is created using plotly.graph_objects. 
# The provided x and y data from the dataframe are added as a trace to the plot. 
# The x-axis and y-axis titles are updated, along with the plot title, height, and width. 
# The plot is then displayed using fig.show(). There is an optional commented-out line that suggests writing the plot to an HTML file.
