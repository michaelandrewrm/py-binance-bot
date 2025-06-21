# feature_engineering.py

import numpy as np

def create_features_labels(df, window):
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = []
    labels = []

    for i in range(len(df) - window - 1):
        feature = df['close'].iloc[i:i+window].values
        label = df['target'].iloc[i+window]
        features.append(feature)
        labels.append(label)

    X = np.array(features)
    y = np.array(labels)
    return X[..., np.newaxis], y
