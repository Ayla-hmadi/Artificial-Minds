

# Step 1: Importing Libraries


import pandas as pd
import numpy as np
import sklearn
import tensorflow
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, Nadam
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


#for data preprocessing
from google.colab import drive

#for loading the model
from tensorflow.keras.models import load_model



# Step 2: Data preperation

#   1.Connecting google drive
drive.mount('/content/drive')

#   2. Load the data into a pandas dataframe
path = '/content/drive/MyDrive/Artifical Minds - CMPS261/HIGGS_train.csv'
data = pd.read_csv(path)

data.info()

#   3. Naming the columns
column_names = ['class_label', 'lepton_pt', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude', 'missing_energy_phi',
                'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag', 'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag',
                'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag', 'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag',
                'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
data.columns = column_names

#   4. Checking for null values
print(data.isna().sum())
data['jet_1_phi'] = pd.to_numeric(data['jet_1_phi'], errors='coerce')
data['jet_4_b-tag'] = pd.to_numeric(data['jet_4_b-tag'], errors='coerce')
data['class_label'] = data['class_label'].astype(int)
data = data.dropna()
data.drop_duplicates(inplace=True)
data.info()




# Step 3: Data Scaling

#   1.  Split the data into features and labels
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
print(y.values)


#   2. Feature scaling"""
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)





#Step 4: Model architecture """

def create_improved_model(activation='relu', optimizer='adam'):
    inputs = Input(shape=(X.shape[1],))
    x = Dense(1024, activation=activation)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Create the model
improved_model = KerasClassifier(build_fn=create_improved_model, epochs=50, batch_size=256 , verbose=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add EarlyStopping and ReduceLROnPlateau callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=0.00001)

# Train and evaluate the model
history = improved_model.fit(X_train, y_train, validation_split=0.33, callbacks=[early_stopping, lr_reduction], verbose=1)
test_accuracy = improved_model.score(X_test, y_test)
print("Test accuracy:", test_accuracy)




#Step 5: Saving, loading and printing the info of our model


# Save the model
improved_model.model.save('my_model.h5')
# Load the saved model
loaded_model = load_model('my_model.h5')
# Print the summary of the loaded model
loaded_model.summary()