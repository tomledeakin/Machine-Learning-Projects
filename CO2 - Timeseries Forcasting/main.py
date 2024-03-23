import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression


def create_ts_data(data, window_size=10, target_size=3):
    count = 1
    while count < window_size:
        data["co2_{}".format(count)] = data["co2"].shift(-count)
        count += 1

    count = 0
    while count < target_size:
        data["target_{}".format(count)] = data["co2"].shift(-count - window_size)
        count += 1
    data = data.dropna(axis=0)
    return data


def main():
    data = pd.read_csv('co2.csv')
    data["time"] = pd.to_datetime((data["time"]))
    data["co2"] = data["co2"].interpolate()

    # print(data.info())
    # fig, ax = plt.subplots()
    # ax.plot(data["time"], data["co2"])
    # ax.set_xlabel("Time")
    # ax.set_ylabel("CO2")
    # plt.show()

    window_size = 5
    target_size = 3
    train_ratio = 0.8
    # data = create_ts_data(data, window_size)
    num_samples = len(data)
    # x = data.drop(["time", "target"], axis=1)
    # y = data["target"]
    # x_train = x[:int(num_samples * train_ratio)]
    # y_train = y[:int(num_samples * train_ratio)]
    # x_test = x[int(num_samples * train_ratio):]
    # y_test = y[int(num_samples * train_ratio):]
    # print(len(x_train), len(x_test))
    #
    # reg = Pipeline(steps=[
    #     ("scaler", StandardScaler()),
    #     ("model", LinearRegression())
    # ])
    #
    # reg.fit(x_train, y_train)
    # y_predict = reg.predict(x_test)
    # mse = mean_squared_error(y_test, y_predict)
    # mae = mean_absolute_error(y_test, y_predict)
    # r2 = r2_score(y_test, y_predict)
    # print("MSE: ", mse)
    # print("MAE: ", mae)
    # print("R2: ", r2)
    #
    # ax.plot(data["time"][:int(num_samples * train_ratio)], y_train, label="train")
    # ax.plot(data["time"][int(num_samples * train_ratio):], y_test, label="test")
    # ax.plot(data["time"][int(num_samples * train_ratio):], y_predict, label="prediction")
    # ax.legend()
    # ax.grid()
    # ax.set_xlabel("Time")
    # ax.set_ylabel("CO2")
    # plt.show()

    # current_data = [380.5, 390, 390.2, 394, 394.4]
    # prediction = reg.predict([current_data])
    # print(prediction)
    # for i in range(10):
    #     prediction = reg.predict([current_data]).tolist()
    #     print("CO2 in week {} is {}".format(i+1, prediction[0]))
    #     current_data = current_data[1:] + prediction

    data = create_ts_data(data, window_size, target_size)

    x = data.drop(["time"] + ["target_{}".format(i) for i in range(target_size)], axis=1)
    y = data[["target_{}".format(i) for i in range(target_size)]]
    print(x.shape, y.shape)
    x_train = x[:int(num_samples * train_ratio)]
    y_train = y[:int(num_samples * train_ratio)]
    x_test = x[int(num_samples * train_ratio):]
    y_test = y[int(num_samples * train_ratio):]

    regs = [LinearRegression() for _ in range(target_size)]
    for i, reg in enumerate(regs):
        reg.fit(x_train, y_train["target_{}".format(i)])

    r2 = []
    mae = []
    mse = []
    for i, reg in enumerate(regs):
        y_pred = reg.predict(x_test)
        r2.append(r2_score(y_test["target_{}".format(i)], y_pred))
        mae.append(mean_absolute_error(y_test["target_{}".format(i)], y_pred))
        mse.append(mean_squared_error(y_test["target_{}".format(i)], y_pred))

    print(r2)
    print(mae)
    print(mse)
    # print(data)


if __name__ == "__main__":
    main()
