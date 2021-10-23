from sklearn.preprocessing import LabelEncoder

def encode_target(df, col, new_col):
    encoder = LabelEncoder()
    df[new_col] = encoder.fit_transform(df[col])
    return encoder, df