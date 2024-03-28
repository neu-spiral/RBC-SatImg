import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Rectangle
from datetime import timedelta


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), ncol=3, bbox_to_anchor=(0.85, 0.69), frameon=False, fontsize=14)


# --------------------------------------------------------------------
# TO BE CHANGED BY USER
# --------------------------------------------------------------------
# TODO: Change scenario accordingly:
scenario_to_plot = 'multiearth'
scenario_to_plot = 'charles_river'

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

timedelta_vector = {'oroville_dam': 5, 'multiearth': 10, 'charles_river': 5}
plt.rc('text', usetex=True)
plt.rc('font', family='times new roman')

if scenario_to_plot == 'multiearth':
    #
    # MultiEarth Timeline
    dates_evaluation = ["2019-07-26", "2019-07-31", "2019-08-10", "2019-08-20",
                        "2019-08-30", "2019-09-09", "2019-10-29", "2020-06-10", "2020-06-15", "2020-06-25",
                        "2020-07-10",
                        "2020-07-15", "2020-07-20", "2020-07-30", "2020-08-04", "2020-08-09", "2020-08-14",
                        "2020-09-03",
                        "2020-10-13", "2020-12-12", "2020-12-22", "2021-01-16", "2021-05-26", "2021-07-25",
                        "2021-07-30",
                        "2021-08-19", "2021-10-13", "2021-12-22"]
    dates_training = ["2019-03-18", "2019-07-11", "2019-07-16"]
    dates_evaluation_labeled = ["2019-08-10", "2020-06-10", "2020-08-04", "2021-05-26", "2021-08-19"]

    #
    show_tags = False

elif scenario_to_plot == 'charles_river':
    #
    # Charles River
    dates_evaluation = pickle.load(open(
        r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/evaluation_results/classification/charles_river_2/date_string_list.pkl",
        'rb'))[2:]
    dates_training = ['2020-09-04', '2020-10-01', '2020-10-09']
    dates_evaluation_labeled = ["2020-11-08", "2020-12-13", "2021-03-20", "2021-04-24", "2021-05-27", "2021-07-31",
                                "2021-09-14"]
    show_tags = False

else:  # scenario_to_plot == 'oroville_dam':
    #
    # Oroville Dam
    dates_evaluation = pickle.load(open(
        r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/evaluation_results/classification/oroville_dam_1/date_string_list.pkl",
        'rb'))
    dates_training = ['2020-09-01', '2020-10-06', '2020-10-11', '2020-10-16']
    dates_evaluation_labeled = ["2020-10-26", "2020-11-25", "2020-12-30", "2021-02-23", "2021-04-09", "2021-05-19",
                                "2021-06-13",
                                "2021-07-08", "2021-08-02", "2021-09-06", "2021-09-26"]
    show_tags = True

# Offset at the start and end of timeline (just for appearance)
if scenario_to_plot == 'multiearth':
    offset = 15
else:
    offset = 5
date_vector = pd.date_range(
    start=(pd.to_datetime([dates_training[0]]) - timedelta(days=offset))[0].strftime('%Y-%m-%d'),
    end=(pd.to_datetime([dates_evaluation[-1]]) + timedelta(days=offset))[0].strftime('%Y-%m-%d'),
    freq="1D")
dates_to_annotate = [dates_training[0], dates_training[-1], dates_evaluation[0], dates_evaluation[-1]]
dates_loop = dates_training + dates_evaluation

# --------------------------------------------------------------------
# Create Figure
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5))

# --------------------------------------------------------------------
# Loop over dates
# --------------------------------------------------------------------

# Annotating only for specific dates and drawing dashed lines
for i, date in enumerate(dates_loop):

    # Get annotation
    date_obj = pd.to_datetime([date])
    date_index = date_vector.get_loc(date_obj[0])
    annotation = date_obj[0].strftime('%Y-%m-%d')[2:]

    # Dashed line linking date annotation and its corresponding marker
    ax.plot([date_vector[date_index], date_vector[date_index]], [0, 0], '--', color='black', lw=1, zorder=1)
    if i < len(dates_training):
        color_i = "green"
        markerfacecolor_i = "white"
        label_i = "Training data"
    elif date in dates_evaluation_labeled:
        color_i = "blue"
        markerfacecolor_i = "blue"
        label_i = "Test data with ground truth"
    else:
        color_i = "blue"
        markerfacecolor_i = "white"
        label_i = "Test data without ground truth"
    line = ax.plot(date_vector[date_index], 0, "-o", color=color_i, markerfacecolor=markerfacecolor_i,
                   zorder=1, label=label_i)
    if date in dates_to_annotate:
        timedelta_i = timedelta_vector[scenario_to_plot]
        if (i + 1) < len(dates_to_annotate):
            index = dates_to_annotate.index(date_obj[0].strftime('%Y-%m-%d'))
            date_obj_next = pd.to_datetime([dates_to_annotate[index + 1]])
            if (date_obj_next - date_obj).days[0] < 20:
                timedelta_i = 15
        ax.annotate(annotation, xy=(date_obj[0] - timedelta(days=timedelta_i), -0.13), xytext=(0, 0),
                    # Adjust the y-offset to move the annotations
                    textcoords='offset points', ha='center', va='top', rotation=45, color='black', fontsize=11)
        ax.plot([date_vector[date_index], date_vector[date_index]], [0, -0.125], '--', color='black',
                lw=1, zorder=1)  # Extend dashed line below the marker

# --------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------
# Plotting the timeline_plot with markers ("-o")
line = ax.plot(date_vector, [0, ] * len(date_vector), zorder=-10,
               color='grey')  # Set zorder higher for the timeline_plot

# Formatting the plot
ax.set_yticks([])  # Hide y-axis ticks and labels
ax.set_xticks([])  # Hide x-axis ticks
ax.set_xticklabels([])  # Hide x-axis tick labels
ax.xaxis.set_tick_params(rotation=70)  # Rotate x-axis labels for better readability
ax.grid(axis='y')  # Show only horizontal gridlines

# Annotating "Training" (centered inside the blue square and above timeline_plot)
training_start_index = date_vector.get_loc(pd.to_datetime([dates_training[0]])[0])
training_end_index = date_vector.get_loc(pd.to_datetime([dates_training[-1]])[0])
training_center = (training_end_index - training_start_index) // 2 + training_start_index

# Highlighting a zone with a rectangle (overlapping the timeline_plot)
start_date = pd.to_datetime([dates_training[0]])
end_date = pd.to_datetime([dates_training[-1]])
highlight_start = date_vector.get_loc(start_date[0])
highlight_end = date_vector.get_loc(end_date[0])
highlight_rect = Rectangle((date_vector[training_start_index], -0.25 / 4),
                           date_vector[training_end_index] - date_vector[training_start_index], 0.25 / 2,
                           edgecolor='green', facecolor='green', alpha=0.1,
                           zorder=2)  # Set zorder lower for the rectangle to overlap the timeline_plot
ax.add_patch(highlight_rect)

# Adding "Evaluation" square
evaluation_start_index = date_vector.get_loc(pd.to_datetime([dates_evaluation[0]])[0])
evaluation_end_index = date_vector.get_loc(pd.to_datetime([dates_evaluation[-1]])[0])
evaluation_rect = Rectangle((date_vector[evaluation_start_index], -0.25 / 4),
                            date_vector[evaluation_end_index] - date_vector[evaluation_start_index], 0.25 / 2,
                            edgecolor='blue', facecolor='blue', alpha=0.04, zorder=2)
ax.add_patch(evaluation_rect)

# Annotating "Evaluation"
evaluation_center = (evaluation_end_index - evaluation_start_index) // 2 + evaluation_start_index

# Removing spines and setting limits
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_visible(False)
ax.set_xlim(date_vector[0], date_vector[-1])
ax.set_ylim(-1, 1)  # Setting the y-axis limits to hide the line
plt.tight_layout()

if show_tags:
    legend_without_duplicate_labels(ax)
    ax.text(date_vector[training_center], 0.1, 'Training', color='green', ha='center', fontsize=14)
    ax.text(date_vector[evaluation_center], 0.1, 'Test data', color='blue', ha='center', fontsize=14)

plt.show()

# --------------------------------------------------------------------
# Save Figure
# --------------------------------------------------------------------
plt.savefig(f"timeline_plot_{scenario_to_plot}.pdf", format='pdf', dpi=1000)
