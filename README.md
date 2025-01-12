# HSLU_DSPRO1_TrafficStatus
HSLU Data Science Project 1 for Traffic Status


## Memebers
Finn Eyer 
Samuel Paul
Tobias Rüedi

## Description
We want to create a heat map of London to show people which areas to avoid at certain times. We also want to provide a pathfinder that shows the user the best route to take.
Therefore we use a dataset called UTD19 (https://utd19.ethz.ch/) and extract the necessary sensor data for the city of London. With the data we compute a regression and make a prediction. 
We then visualize the obtained information with a grid over London.

# Dashboard
The following steps are required to use the dashboard itself
First, the Github repository must be cloned.
Then the zip file dataset_version1.0_22.11.2024 must be downloaded and unpacked under data/ dataset_V1.0/. The path to the contained files must be adjusted accordingly under grid/grid_functions.py in the following functions.
- get_weekday_prediction
- plot_detections_as_points

The required package should then be installed with the requirements.txt file. It should then be possible to start the dashboard.
The start takes about 45 minutes, as all models are first queried and the results are written to a cashe to ensure smooth use of the dashboard later.
If this is not desired, line 168 in Dashboard.py can be changed as a comment, then the dashboard is started without prior cashing.