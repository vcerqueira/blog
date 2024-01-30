from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier
from scipy.stats import ks_2samp

stream = SEAGenerator(classification_function=2,
                      random_state=112,
                      balance_classes=False,
                      noise_percentage=0.28)

X, y = stream.next_sample(1000)

model = AdaptiveRandomForestClassifier()

model.fit(X, y)

fixed_reference_x1 = X[:,0]

buffer_x1 = []

# processing each instance of the data stream
while stream.has_more_samples():
    # getting a new sample
    X_i, y_i = stream.next_sample()
    # making the prediction
    pred = model.predict(X_i)

    # monitoring the variable x1

    ## adding x1 value to buffer
    buffer_x1.append(X_i[0][0])

    if len(buffer_x1) > 1050:
        break

    ## getting the detection window (latest 1000 records)
    detection_window = buffer_x1[-1000:]

    ## using KS test
    test = ks_2samp(fixed_reference_x1, detection_window)

    ## checking if change occurs (pvalue is below 0.001)
    change_detected = test.pvalue < 0.001

    if change_detected:
        print('Update model')

