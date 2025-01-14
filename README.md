# HSLU_DSPRO1_TrafficStatus
HSLU Data Science Project 1 for Traffic Status

## Memebers
Finn Eyer </br>
Samuel Paul </br>
Tobias Rüedi </br>

## Disclaimer
Most Jupyter notebooks are not executable due to file or library imports. For testing, please run our Python scripts like create_dataset.py test_train_split_function.py
Our scientific report, Traffic Status, includes a detailed description of why this happens. Please use the Python Version python == 3.13.0.

## Description
We want to create a heat map of London to show people which areas to avoid at certain times. We also want to provide a pathfinder that shows the user the best route to take.
Therefore we use a dataset called UTD19 (https://utd19.ethz.ch/) and extract the necessary sensor data for the city of London. With the data, we compute a regression and make a prediction. 
We then visualize the obtained information with a grid over London.

## Dashboard
If you want to use the dashboard, the following steps are necessary.
First clone the Github repository.
Install Python version 3.13.0 and install the required libaries from the requirements.txt file.
Now it can already be started by executing the Dashboard.py.

If you do not want to use the cached predictions, the following steps must be completed:
Then download the zip file dataset_version1.0_22.11.2024 and unpack it under data/ dataset_V1.0/. Change the path to the files contained in grid/grid_functions.py in the following functions.
- get_weekday_prediction
- plot_detectors_as_points
Then delete the cached files under /grid/chache-directory. The new files will then be created when you start the application. Please note that this will take about 45 minutes.

### Example
Once launched, the Dashboard can be accessed via http://127.0.0.1:8050/.
Here you can choose between the different models, the default being our KNN model.
You can also select the mode, the day of the week and the time of day for your preferred forecast.

![Demo_Dashboard](src/image.png)

