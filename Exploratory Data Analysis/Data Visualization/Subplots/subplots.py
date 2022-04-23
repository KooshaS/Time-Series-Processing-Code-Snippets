# Ref: https://www.kaggle.com/code/tyrionlannisterlzy/xgboost-dnn-ensemble-lb-0-980/notebook

import matplotlib.pyplot as plt

sequences = [0, 1, 2, 3, 4, 5] #selects sequences for display
number_of_sensors = 13
figure, axes = plt.subplots(number_of_sensors, len(sequences), sharex=True, figsize=(16, 16))
for i, sequence in enumerate(sequences):
    for sensor in range(number_of_sensors):
        sensor_name = f"sensor_{sensor:02d}"
        plt.subplot(number_of_sensors, len(sequences), sensor * len(sequences) + i + 1)
        plt.plot(range(60), train[train.sequence == sequence][sensor_name],
                color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10])
        if sensor == 0: plt.title(f"Sequence {sequence}")
        if sequence == sequences[0]: plt.ylabel(sensor_name)
figure.tight_layout(w_pad=0.1)
plt.suptitle('Selected Time Series', y=1.02)
plt.show()
