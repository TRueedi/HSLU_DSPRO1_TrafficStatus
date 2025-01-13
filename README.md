# HSLU_DSPRO1_TrafficStatus
HSLU Data Science Project 1 for Traffic Status

## Memebers
Finn Eyer </br>
Samuel Paul </br>
Tobias Rüedi </br>

## Disclaimer
Most Jupyter notebooks are not executable due to file or library imports. For testing, please run our Python scripts like create_dataset.py test_train_split_function.py
Our scientific report, Traffic Status, includes a detailed description of why this happens.

## Description
We want to create a heat map of London to show people which areas to avoid at certain times. We also want to provide a pathfinder that shows the user the best route to take.
Therefore we use a dataset called UTD19 (https://utd19.ethz.ch/) and extract the necessary sensor data for the city of London. With the data, we compute a regression and make a prediction. 
We then visualize the obtained information with a grid over London.

## Dashboard
To use the dashboard itself, the following steps are required
First, clone the Github repository.
Then download the zip file dataset_version1.0_22.11.2024 and unzip it under data/ dataset_V1.0/. Change the path to the files it contains in grid/grid_functions.py in the following functions.
- get_weekday_prediction
- plot_detections_as_points

The required package should then be installed using the requirements.txt file. You should then be able to start the dashboard.
At the moment the prechached prediction is loaded, if new ones are needed, please delete the files in /grid/chache-directory. New ones will then be created when the application starts, but please note that this will take about 45 minutes.

### Example
Once launched, the Dashboard can be accessed via http://127.0.0.1:8050/.
Here you can choose between the different models, the default being our KNN model.
You can also select the mode, the day of the week and the time of day for your preferred forecast.

![Demo_Dashboard](src/image.png)

