import pandas as pd
import STRING

def train_test_valid_data(data, train_path, valid_path, test_path):
    df = pd.read_csv(data, sep=';', encoding='latin1')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE', ascending=True)
    df['AR7'] = df['PASSENGER_SUM_DAY'].shift(7)
    df = df.dropna(axis=0)

    test = df[df['DATE'] >= STRING.fares.date_fare_apr_16]
    df = df[df['DATE'] < STRING.fares.date_fare_apr_16]
    df = df.reset_index(drop=True)

    split = int(len(df.index)*0.80)
    train = df.loc[0:split + 1]
    valid = df.loc[split + 1:]

    train.to_csv(train_path, sep=';', index=False, encoding='latin1')
    valid.to_csv(valid_path, sep=';', index=False, encoding='latin1')
    test.to_csv(test_path, sep=';', index=False, encoding='latin1')


if __name__ == '__main__':
    train_test_valid_data(STRING.temporal_data, STRING.train, STRING.valid, STRING.test)
    train_test_valid_data(STRING.temporal_data_detrend, STRING.train_detr, STRING.valid_detr, STRING.test_detr)