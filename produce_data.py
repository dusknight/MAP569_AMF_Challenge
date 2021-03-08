import pandas as pd


def additional_training_data(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    y_train.append(y_test)
    for k in y_test.Trader:
        x_train.append(x_test[x_test['Trader'] == k])
    return x_train, y_train


if __name__ == "__main__":
    x_test = pd.read_csv('data/Agmt_test_X.csv')
    x_train = pd.read_csv('data/Agmt_train_X.csv')
    y_train = pd.read_csv('data/AMF_train_Y.csv')
    y_test = pd.read_csv('data/AMF_test_Y.csv')
    x_train, y_train = additional_training_data(
        x_train, x_test, y_train, y_test)
    x_train.to_csv('data/Agmted_train_X.csv', index=False)
    y_train.to_csv('data/Agmted_train_Y.csv', index=False)
