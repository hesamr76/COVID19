import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping

country = "Iran"

# declare directories
dataset_dir = "dataset/confirmed.csv"

# load data from csv
dataset = pd.read_csv(dataset_dir)

# select target country
confirmed = dataset[dataset["Country/Region"] == country]

# cumulative datas
cumulative_confirmed = pd.DataFrame(
    confirmed[confirmed.columns[4:]].sum(), columns=["Confirmed"])

# normalize dates of dataset
cumulative_confirmed.index = pd.to_datetime(
    cumulative_confirmed.index, format="%m/%d/%y")
cumulative_confirmed.index.name = country

# i want to predict 7 days afterwards
goal = 7
train_size = len(cumulative_confirmed) - goal

# split test data from data set
train_data = cumulative_confirmed[:train_size]
test_data = cumulative_confirmed[train_size:]

# normalize the dataset
scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train = scaler.transform(train_data)
scaled_test = scaler.transform(test_data)

# create generator
generator = TimeseriesGenerator(
    scaled_train, scaled_train, length=goal, batch_size=1)

# create model
model = Sequential()
model.add(LSTM(75, activation="relu", input_shape=(goal, 1)))
model.add(Dense(14, activation="relu"))
model.add(Dense(units=1))
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

# create validation data
validation_data = np.append(
    scaled_train[train_size - 1], scaled_test)
validation_data = validation_data.reshape(goal + 1, 1)
validation_data = TimeseriesGenerator(
    validation_data, validation_data, length=goal, batch_size=1)

# fit model with early stop
early_stop = EarlyStopping(
    monitor="val_loss", patience=25, restore_best_weights=True)

model.fit_generator(generator, validation_data=validation_data,
                    epochs=100, callbacks=[early_stop], steps_per_epoch=goal * 2)


# save the model
model.save("model.h5")
print("Saved model")

# declare predictions array
scaled_predictions = []

# initial batch
init_batch = scaled_train[-goal:]
current_batch = init_batch.reshape(1, goal, 1)

# get predictions and fill scaled_predictions
for i in range(goal):
    current_prediction = model.predict(current_batch)[0]
    scaled_predictions.append(current_prediction)
    current_batch = np.append(current_batch[:, 1:, :], [
                              [current_prediction]], axis=1)

# invert minmax scale
predictions = scaler.inverse_transform(scaled_predictions)

# create output
time_series = test_data.index
output = pd.DataFrame(columns=["Confirmed", "Predicted"], index=time_series)
output["Predicted"] = predictions
output["Confirmed"] = test_data["Confirmed"]

err = np.mean(np.abs(np.array(
    output["Confirmed"]) - np.array(output["Predicted"]))/np.array(output["Confirmed"]))
accuracy = round((1 - err) * 100, 2)
accuracy = " (accuracy: " + str(accuracy) + "%)"

# print output
print(output)
print(accuracy)

# show plot
pd.plotting.register_matplotlib_converters()

plt.plot(output)
plt.legend(["Confirmed cases", "Predicted cases"])
plt.title(country + accuracy)
plt.xlabel("Date")
plt.ylabel("Confirmed")
figure = plt.gcf()
ax = plt.gca()
date_format = DateFormatter("%b/%d")
ax.xaxis.set_major_formatter(date_format)
plt.show()
figure.savefig('outputs/' + country + '.jpg', dpi=100)
