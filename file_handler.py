import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures  # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression, \
    HuberRegressor  # classes providing Linear Regression with ordinary squared error loss and Huber loss, respectively
from sklearn.metrics import mean_squared_error, mean_absolute_error  # function to calculate mean squared error


def csv_handler():
    # this reads the csv file, for now it uses pandas, use RESTapi to get the file for future.
    # Deal with different header naming conventions.
    df = pd.read_csv("rsrp_vs_pdcp_DL_mob.csv", sep=';', engine='python')
    df = pd.DataFrame(df)
    df.dropna()
    PDCP = df['PDCP DL bitrate'].str.replace(',', '.').astype(float)
    RSRP = df['1. best RSRP'].str.replace(',', '.').astype(float)
    # rsrp_pdcp = df[['PDCP DL bitrate', '1. best RSRP']].str.apply(lambda x: x.replace(',', '.')).astype('float')
    list_of_tuples = list(zip(PDCP, RSRP))
    df = pd.DataFrame(list_of_tuples,
                      columns=['PDCP', 'RSRP'])
    return df


def ssBlockPower_input():
    while True:
        ssblockPower = int(input("Enter the sspowerBlock(dB): "))
        if ssblockPower > 0:
            print("Ssblock Power is a negative number")
        else:
            return ssblockPower


def pathloss_calculation(rsrp, ssblockPower):
    # Use to calculate the pathloss = rsrp - ssblockpower
    global pathloss
    data = []
    for i in rsrp:
        pathloss = i - ssblockPower
        data.append(pathloss)
    pathloss = pd.DataFrame(data, columns=['Pathloss'])
    # print(pathloss)
    return pathloss


def plot_function(pathloss, pdcp):
    pathloss.to_numpy()
    pdcp.to_numpy()
    # print(pathloss)
    # print(pdcp)
    x = pathloss
    y = pdcp
    plt.scatter(x, y)
    plt.gca().invert_xaxis()
    plt.xlabel("Pathloss (dB)")
    plt.ylabel("Throughput (Mbps)")
    plt.show()


def machine_learning(pathloss, pdcp):
    # feature X = pathloss
    # label Y = pdcp
    ## 1. Fit a linear regression model
    X = pathloss.to_numpy().reshape(-1, 1)
    y = pdcp.to_numpy()

    y = np.where(np.isnan(y), 0, y)

    regr = LinearRegression()
    regr.fit(X, y)
    ## 2.Predict label values based on features and calculate the training error
    y_pred = regr.predict(X)
    tr_error = mean_absolute_error(y, y_pred)
    print('The training error is: ', tr_error)  # print the training error
    print("w1 = ", regr.coef_)  # print the learnt w1
    print("w0 = ", regr.intercept_)  # print the learnt w0
    ## visualize the model you have learnt, you are supposed to see the datapoints and the fitted h(x), a straignt line

    plt.figure(figsize=(8, 6))  # create a new figure with size 8*6

    # create a scatter plot of datapoints
    # each datapoint is depicted by a dot in color 'blue' and size '10'
    plt.scatter(X, y, color='b', s=8, label='datapoints from the dataframe')

    # plot the predictions obtained by the learnt linear hypothesis using color 'red' and label the curve as "h(x)"
    y_pred = regr.predict(X)  # predict using the linear model
    plt.plot(X, y_pred, color='r', label='h(x)')
    plt.gca().invert_xaxis()
    plt.xlabel('Pathloss (dB)', size=15)  # define label for the horizontal axis
    plt.ylabel('Throughput (Mbps)', size=15)  # define label for the vertical axis

    plt.title('Linear regression model', size=15)  # define the title of the plot
    plt.legend(loc='best', fontsize=14)  # define the location of the legend

    plt.show()  # display the plot on the screen
    ## define a list of values for polynomial degrees
    degrees = [3, 5, 10]

    # declare a variable to store the resulting training errors for each polynomial degree
    tr_errors = []

    for i in range(len(degrees)):  # use for-loop to fit polynomial regression models with different degrees

        print("Polynomial degree = ", degrees[i])

        poly = PolynomialFeatures(degree=degrees[i])  # initialize a polynomial feature transformer
        X_poly = poly.fit_transform(X)  # fit and transform the raw features

        lin_regr = LinearRegression(
            fit_intercept=False)  # NOTE: "fit_intercept=False" as we already have a constant iterm in the new feature X_poly
        lin_regr.fit(X_poly, y)  # fit linear regression to these new features and labels (labels remain same)

        y_pred = lin_regr.predict(X_poly)  # predict using the learnt linear model
        tr_error = mean_absolute_error(y, y_pred)  # calculate the training error

        # sanity check the feature matrix is transformed correctly
        assert X_poly.shape == (X.shape[0], degrees[i] + 1), "Wrong polynomial degree"

        print("The first two row of X_poly: \n", X_poly[0:2])

        print("\nThe learned weights: \n", lin_regr.coef_)

        tr_errors.append(tr_error)
        X_fit = np.linspace(-55, -95, 350)  # generate samples
        plt.plot(X_fit, lin_regr.predict(poly.transform(X_fit.reshape(-1, 1))),
                 label="Model")  # plot the polynomial regression model
        plt.gca().invert_xaxis()
        plt.scatter(X, y, color="b", s=10,
                    label="datapoints from the dataframe ")  # plot a scatter plot of y(throughput) vs. X(pathlosss) with color 'blue' and size '10'
        plt.xlabel('Pathloss (dB)')  # set the label for the x/y-axis
        plt.ylabel('Throughput (Mbps)')
        plt.legend(loc="best")  # set the location of the legend
        plt.title('Polynomial degree = {}\nTraining error = {:.5}'.format(degrees[i], tr_error))  # set the title
        plt.show()  # show the plot


rsrp_pdcp = csv_handler()

ssbP = ssBlockPower_input()

pdcp = rsrp_pdcp['PDCP']

pathloss = pathloss_calculation(rsrp_pdcp['RSRP'], ssbP)

plot_function(pathloss, pdcp)
machine_learning(pathloss, pdcp)
