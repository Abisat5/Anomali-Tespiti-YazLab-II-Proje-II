import numpy as np


def create_sequences(feature_matrix, labels, seq_len):
    """
    Zaman serisi verisinden (n_timesteps, n_features) 3D tensör üretir.
    Her pencere için etiket, penceredeki maksimum anomali değeri olarak alınır.
    """
    if len(feature_matrix) < seq_len:
        return np.empty((0, seq_len, feature_matrix.shape[1])), np.empty((0,))

    sequences = []
    sequence_labels = []

    for start in range(len(feature_matrix) - seq_len + 1):
        end = start + seq_len
        sequences.append(feature_matrix[start:end])
        sequence_labels.append(int(np.max(labels[start:end])))

    return np.array(sequences), np.array(sequence_labels)


def reconstruction_scores(model, x_data):
    predictions = model.predict(x_data, verbose=0)
    return np.mean(np.power(x_data - predictions, 2), axis=(1, 2))


def threshold_from_validation(model, x_val, percentile):
    scores = reconstruction_scores(model, x_val)
    if len(scores) == 0:
        return 0.0
    return float(np.percentile(scores, percentile))


def predict_with_threshold(model, x_test, threshold):
    scores = reconstruction_scores(model, x_test)
    y_pred = (scores > threshold).astype(int)
    return y_pred, scores
