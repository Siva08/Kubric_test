import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"
response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    ...


def predict_price(area) -> float:
    train_data = pandas.DataFrame(pandas.read_csv(TRAIN_DATA_URL))
    xtrain = train_data.iloc[:,0:-1]
    ytrain = train_data.iloc[:,-1]
    "Feature Scalling"
    xtrain = (xtrain - xtrain.mean)/(xtrain.max() - xtrain.min())
    xtrain = xtrain.to_numpy()
                   _
    "Analytical solution"
    C = 0.2
    rate = 0.5
    total_step = 500
    """weights = numpy.dot(numpy.transpose(X), X)
    weights = numpy.linalg.inv((weights + C*numpy.eye(numpy.shape(xtrain)[1])))
    weights  = numpy.dot(weights,np.dot(np.transpose(X),y))
    weights = np.dot(weights,ytrain)
    """
    newWeigths = numpy.random.rand(numpy.shape(xtrain)[1])
    
    "Prediction"
    predictions = numpy.dot(xtrain,weights)
    "Gardient descent"
    
    difference = predictions - ytrian
    MSE = numpy.sum(difference*diffence)/len(ytrain)
    L2 = numpy.sum(wieghts*weights)
    Loss = MSE+C*L2
    gradients  = numpy.dot(numpy.transpose(xtrain), difference)/len(ytrain)
    newWeighs =  newWeigths - gradients*rate
    for step in range(total_step):
        gradients  = numpy.dot(numpy.transpose(xtrain), difference)/len(ytrain)
        newWeighs =  newWeigths - gradients*rate
        if step%10:
            MSE = numpy.sum(difference*diffence)/len(ytrain)
            L2 = numpy.sum(wieghts*weights)
            Loss = MSE+C*L2
            print("step : "+ str(step) + "loss" + str(Loss))
    
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
 

if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
