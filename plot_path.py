import numpy as np
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

def visualize_paths(pred_path, html_title="", file_out="plot.html"):
    output_file(file_out, title=html_title)

    pred_path = np.array(pred_path)  # Convert to NumPy array if not already done

    # Extract x and y coordinates
    pred_x, pred_y = pred_path.T

    # Create ColumnDataSource with only path data
    source = ColumnDataSource(data=dict(px=pred_x, py=pred_y))

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"
    
    # Create the figure for paths
    fig = figure(title="Paths", tools=tools, match_aspect=True, width_policy="max",
                 toolbar_location="above", x_axis_label="x", y_axis_label="y")
    
    # Plot the paths using circles and lines
    fig.circle("px", "py", source=source, color="green", legend_label="Paths")
    fig.line("px", "py", source=source, color="green", legend_label="Paths")
    
    # Set the legend interaction policy
    fig.legend.click_policy = "hide"

    # Show the plot
    show(fig)


def visualize_paths_with_target(pred_path, possible_targets, html_title="", file_out="plot.html"):
    output_file(file_out, title=html_title)

    pred_path = np.array(pred_path)  # Convert to NumPy array if not already done

    # Extract x and y coordinates
    pred_x, pred_y = pred_path.T

    target_x = [tar[0] for tar in possible_targets]
    target_y = [tar[1] for tar in possible_targets]

    # Create ColumnDataSource with only path data
    source = ColumnDataSource(data=dict(px=pred_x, py=pred_y))

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"
    
    # Create the figure for paths
    fig = figure(title="Paths", tools=tools, match_aspect=True, width_policy="max",
                 toolbar_location="above", x_axis_label="x", y_axis_label="y")
    
    # Plot the paths using circles and lines
    fig.circle("px", "py", source=source, color="green", legend_label="Paths")
    fig.line("px", "py", source=source, color="green", legend_label="Paths")

    # print("Adding target location: {}, {}".format(target_loc[0], target_loc[1]))
    # fig.circle(x=[target_loc[0]], y=[target_loc[1]], color="red", legend_label="Target", size=15)
    fig.circle(x=target_x, y=target_y, color="red", legend_label="Target", size=15)
    
    # Set the legend interaction policy
    fig.legend.click_policy = "hide"

    # Show the plot
    show(fig)

# Example usage:
# Assuming pred_path is your data containing the paths
# visualize_paths(pred_path)
