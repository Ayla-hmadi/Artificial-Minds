# HIGGS Boson Machine Learning Model

## Problem Statement
The field of high-energy physics is devoted to the study of the elementary constituents of matter. By investigating the structure of matter and the laws that govern its interactions, this field strivesm to discover the fundamental properties of the physical universe. The primary tools of experimental high-energy physicists are modern accelerators, which collide protons and/or antiprotons to create exotic particles that occur only at extremely high-energy densities. Observing these particles and measuring their properties may yield critical insights about the very nature of matter. Collisions at high-energy particle colliders are a traditionally fruitful source of exotic particle discoveries. Finding these rare particles requires solving difficult signal-versus-background classification problems, hence machine-learning approaches are often used. Given the limited quantity and expensive nature of the data, improvements in analytical tools directly boost particle discovery potential. The vast majority of particle collisions do not produce exotic particles. For example, though the Large Hadron Collider (LHC) produces approximately 1011 collisions per hour, approximately 300 of these collisions result in a Higgs boson, on average. Therefore, good data analysis depends on distinguishing collisions which produce particles of interest (signal) from those producing other particles (background).

## Solution
In this project, we developed a machine learning model with the goal to distinguish between a signal process where new theoretical Higgs bosons (HIGGS) are produced, and a background process with the identical decay products but distinct kinematic features.

## Dataset
Each process (signal or background) in the dataset is represented by 28 features. The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes. There is an interest in using deep learning methods to obviate the need for physicists to manually develop such features. The first column is the class label (1 for signal, 0 for background), followed by the 28 features (21 low-level features then 7 high-level features): lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m jj, m jjj, m lv, m jlv, m bb, m wbb, m wwbb.

## Dependencies
To run the code in this repository, you will need the following libraries:
- pandas
- numpy
- sklearn
- tensorflow
- keras (specifically keras.wrappers.scikit_learn)

You can install these with pip:
````pip install pandas numpy sklearn tensorflow keras````

## Data Preparation
The code reads data from a CSV file stored on a Google Drive, pre-processes the data, and loads it into a pandas dataframe. The data is then cleaned, removing any duplicates or null values.

## Model Architecture
The neural network model is defined in the create_improved_model function. This model includes multiple dense layers, batch normalization, and dropout layers. The model can be compiled with any activation function and optimizer specified when calling the function.

## Training and Evaluation
After defining the model, the code splits the data into training and testing sets. It trains the model using the training data, and then evaluates it using the test data. It also applies the EarlyStopping and ReduceLROnPlateau callbacks during training.
After training and evaluating the model, the code saves the model to a file and can load the model from the saved file.

## Instructions to run
Make sure to replace the Google Drive path to CSV data file with the path where your data file is located.
````path = '/content/drive/MyDrive/YourPath/YourData.csv'
data = pd.read_csv(path)````

**Note**: This code runs in a Google Colab environment and requires Google Drive for accessing the data. Make sure you have access to Google Colab and Google Drive. If you want to run the code in a different environment or use a different data source, you may need to modify the data loading and processing code.

## Contact
For any queries, please feel free to reach out :)
