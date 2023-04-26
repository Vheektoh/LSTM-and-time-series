from copy import deepcopy as dc

def prepare_dataframe_for_lstm(df, n_steps):
    '''takes a dataframe and returns a shifted dataframe with the number of steps as lag'''

    df = dc(df)

    df.set_index('Datetime', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Hourly_Temp(t - {i})'] = df['Hourly_Temp'].shift(i)
    df.dropna(inplace=True)

    return df