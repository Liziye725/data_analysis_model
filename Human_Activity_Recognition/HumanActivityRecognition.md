# Human Activity Recognition
Human Activity Recognition (HAR) using smartphone sensors such as accelerometers and gyroscopes. HAR is a type of time series classification problem where various machine learning and deep learning models are employed to achieve optimal results. This project utilizes the Long Short-Term Memory (LSTM) model, a variant of Recurrent Neural Networks (RNN), to recognize different human activities like standing, climbing upstairs, and downstairs.

## Objective

The primary goal of this project is to recognize human activities using data from smartphone sensors. This involves leveraging the capabilities of LSTM networks to capture order dependencies in sequence prediction tasks.


## Activities Recognized

The activities classified in this project include:

- Walking
- Upstairs
- Downstairs
- Sitting
- Standing

## Dataset

The dataset for this project can be downloaded from the following link: [HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

## Understanding the Dataset

- **Sensor Data:** The accelerometers and gyroscopes generate data in 3D space over time.
  - *Accelerometer:* Measures magnitude and direction of proper acceleration.
  - *Gyroscope:* Maintains orientation based on the conservation of angular momentum.
- **Data Representation:** The data is represented in 3-axial signals (X, Y, Z).
- **Preprocessing:** The data is pre-processed using noise filters and sampled into fixed-width windows of 128 readings each.
- **Data Split:** 80% of the volunteers' data is used for training, and the remaining 20% is used for testing.

## Project Phases

### Phase 1: Dataset Selection

- Download the HAR dataset from the provided link.

### Phase 2: Data Upload and Setup

- Upload the dataset to Google Drive to work with Google Colaboratory.

### Phase 3: Data Cleaning and Preprocessing

- Apply necessary preprocessing steps to clean the data and prepare it for model training.

### Phase 4: Model Selection and Building

- Choose an appropriate machine learning or deep learning model.
- Build a deep learning network model using LSTM.

### Phase 5: Exporting the Model

- Export the trained model for integration with Android Studio for further application development.

## Implementation Details

### Tools and Environment

- **IDE:** Google Colaboratory is used for its efficient handling of deep learning projects.
- **Libraries:** Import all necessary libraries in a new notebook in Google Colaboratory to start the project.

### Instructions

1. **Open Google Colaboratory:** Create a new notebook.
2. **Import Libraries:** Import essential libraries for data processing, model building, and evaluation.
3. **Load Data:** Load the HAR dataset from Google Drive.
4. **Preprocess Data:** Apply cleaning and preprocessing techniques.
5. **Build Model:** Develop an LSTM model to recognize human activities.
6. **Train Model:** Train the model using the training dataset.
7. **Evaluate Model:** Evaluate the model performance using the test dataset.
8. **Export Model:** Save and export the model for use in Android Studio.