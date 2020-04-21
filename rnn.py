import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping

country = "Iran"

# declare directories
# confirmed_dir = "dataset/confirmed.csv"
confirmed_dir = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

# load data from csv
confirmed_data = pd.read_csv(confirmed_dir)

# select target country
confirmed = confirmed_data[confirmed_data["Country/Region"] == country]

# cumulative datas
cumulative_confirmed = pd.DataFrame(
    confirmed[confirmed.columns[4:]].sum(), columns=["Confirmed"])

# normalize dates of dataset
cumulative_confirmed.index = pd.to_datetime(
    cumulative_confirmed.index, format="%m/%d/%y")
cumulative_confirmed.index.name = country

# i want to predict 7 days afterwards
goal = 7
confirmed_train_size = len(cumulative_confirmed) - goal

# split test data from data set
confirmed_train_data = cumulative_confirmed[:confirmed_train_size]
confirmed_test_data = cumulative_confirmed[confirmed_train_size:]

# normalize the dataset
scaler = MinMaxScaler()
scaler.fit(confirmed_train_data)
scaled_confirmed_train = scaler.transform(confirmed_train_data)
scaled_confirmed_test = scaler.transform(confirmed_test_data)

# create generator
generator = TimeseriesGenerator(
    scaled_confirmed_train, scaled_confirmed_train, length=goal, batch_size=1)

# create model
model = Sequential()
model.add(LSTM(75, activation="relu", input_shape=(goal, 1)))
model.add(Dense(14, activation="relu"))
model.add(Dense(units=1))
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

# create validation data
confirmed_validation = np.append(
    scaled_confirmed_train[confirmed_train_size - 1], scaled_confirmed_test)
confirmed_validation = confirmed_validation.reshape(goal + 1, 1)
confirmed_validation_gen = TimeseriesGenerator(
    confirmed_validation, confirmed_validation, length=goal, batch_size=1)

# fit model with early stop
early_stop = EarlyStopping(
    monitor="val_loss", patience=25, restore_best_weights=True)

model.fit_generator(generator, validation_data=confirmed_validation_gen,
                    epochs=100, callbacks=[early_stop], steps_per_epoch=goal * 2)


# save the model
model.save("model.h5")
print("Saved model")

# declare predictions array
scaled_predictions = []

# initial batch
init_batch = scaled_confirmed_train[-goal:]
current_batch = init_batch.reshape(1, goal, 1)

# get predictions and fill scaled_predictions
for i in range(goal * 2):
    current_prediction = model.predict(current_batch)[0]
    scaled_predictions.append(current_prediction)
    current_batch = np.append(current_batch[:, 1:, :], [
                              [current_prediction]], axis=1)

# invert minmax scale
predictions = scaler.inverse_transform(scaled_predictions)

# extend data for goal (7) next days
time_series = confirmed_test_data.index
for k in range(0, goal):
    time_series = time_series.append(
        time_series[-1:] + pd.DateOffset(1))

output = pd.DataFrame(
    columns=["Confirmed", "Predicted"], index=time_series)
output["Predicted"] = predictions
output["Confirmed"] = confirmed_test_data["Confirmed"]

print(output)


def showPlot(data):
    pd.plotting.register_matplotlib_converters()

    plt.plot(data)
    plt.legend(["Confirmed cases", "Predicted cases"])
    plt.title(country)
    plt.xlabel("Date")
    plt.ylabel("Confirmed")
    figure = plt.gcf()
    plt.show()
    figure.savefig('outputs/' + country + '.jpg', dpi=100)


showPlot(output)
