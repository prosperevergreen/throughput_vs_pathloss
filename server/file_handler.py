from __future__ import division
from numpy import ones, convolve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures  # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression, \
    HuberRegressor  # classes providing Linear Regression with ordinary squared error loss and Huber loss, respectively
from sklearn.metrics import mean_squared_error, mean_absolute_error  # function to calculate mean squared error


class Processor:
    ssblock_power = 0

    def __init__(self, df):
        self.df = df
        # self.df2 = df2
        # self.rsrp_ = []
        self.pdcp_ = []
        self.pathloss = []

    def csv_handler(self):
        # this reads the csv file, for now it uses pandas, use RESTapi to get the file for future.
        # Deal with different header naming conventions.
        # df = pd.read_csv("rsrp_vs_pdcp_DL_mob.csv", sep=';', engine='python')
        df = pd.DataFrame(self.df)
        df.dropna()
        # PDCP = df['PDCP DL bitrate'].str.replace(',', '.').astype(float)
        # RSRP = df['1. best RSRP'].str.replace(',', '.').astype(float)
        # rsrp_pdcp = df[['PDCP DL bitrate', '1. best RSRP']].str.apply(lambda x: x.replace(',', '.')).astype('float')
        # list_of_tuples = list(zip(PDCP, RSRP))
        # df = pd.DataFrame(list_of_tuples,
        #                 columns=['PDCP', 'RSRP'])
        print(df)
        self.pdcp_ = df['pdcp']
        self.pathloss = df['pathloss']
        return self.pdcp_, self.pathloss

    # def dataFrame_handler(self):
    #     # df = pd.DataFrame(df)
    #     self.df.dropna()
    #     PDCP = self.df['PDCP DL bitrate'].str.replace(',', '.').astype(float)
    #     RSRP = self.df['1. best RSRP'].str.replace(',', '.').astype(float)
    #     # rsrp_pdcp = df[['PDCP DL bitrate', '1. best RSRP']].str.apply(lambda x: x.replace(',', '.')).astype('float')
    #     list_of_tuples = list(zip(PDCP, RSRP))
    #     df = pd.DataFrame(list_of_tuples,
    #                       columns=['PDCP', 'RSRP'])
    #     print(df)
    #     return df

    # def ssBlockPower_input(self):
    #     while True:
    #         ssblockPower = int(input("Enter the sspowerBlock(dB): "))
    #         if ssblockPower > 0:
    #             print("Ssblock Power is a negative number")
    #         else:
    #             return ssblockPower

    # def pathloss_calculation(self, rsrp, ssblock_power):
    #     # Use to calculate the pathloss = rsrp - ssblockpower
    #     global pathloss
    #     data = []
    #     for i in rsrp:
    #         pathloss = i - ssblock_power
    #         data.append(pathloss)
    #     pathloss = pd.DataFrame(data, columns=['Pathloss'])
    #     # print(pathloss)
    #     self.pathloss = pathloss
    #     return pathloss

    # def plot_function(self, pathloss, pdcp):
    #     # print(pathloss)
    #     # print(pdcp)
    #     x = pathloss
    #     y = pdcp
    #     plt.scatter(x, y)
    #     plt.gca().invert_xaxis()
    #     plt.xlabel("Pathloss (dB)")
    #     plt.ylabel("Throughput (Mbps)")
    #     plt.show()

    def machine_learning(self, pathloss, pdcp):
        # feature X = pathloss
        # label Y = pdcp
        pathloss = pd.DataFrame(pathloss)
        pdcp = pd.DataFrame(pdcp)
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
        #return y_pred

        for i in range(len(degrees)):  # use for-loop to fit polynomial regression models with different degrees

            print("Polynomial degree = ", degrees[i])

            poly = PolynomialFeatures(degree=degrees[i])  # initialize a polynomial feature transformer
            X_poly = poly.fit_transform(X)  # fit and transform the raw features

            lin_regr = LinearRegression(
                fit_intercept=False)  # NOTE: "fit_intercept=False" as we already have a constant iterm in the new feature X_poly
            lin_regr.fit(X_poly, y)  # fit linear regression to these new features and labels (labels remain same)

            y_pred = lin_regr.predict(X_poly)  # predict using the learnt linear model
            tr_error = mean_absolute_error(y, y_pred)  # calculate the training error

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

    # rsrp_pdcp = csv_handler()

    # ssbP = ssBlockPower_input()

    # pdcp = rsrp_pdcp['PDCP']

    # pathloss = pathloss_calculation(rsrp_pdcp['RSRP'], ssbP)

    # plot_function(pathloss, pdcp)
    # machine_learning(pathloss, pdcp)


class Plotter:
    def __init__(self, pdcp1, pathloss1, pdcp2, pathloss2):
        self.pdcp1 = pdcp1
        self.pathloss1 = pathloss1
        self.pdcp2 = pdcp2
        self.pathloss2 = pathloss2

    # def pathloss_calculation_dataset_one(self, rsrp):
    #     # Use to calculate the pathloss = rsrp - ssblockpower
    #     global pathloss
    #     data = []
    #     for i in rsrp:
    #         pathloss = i - self.sspower
    #         data.append(pathloss)
    #     pathloss = pd.DataFrame(data, columns=['Pathloss'])
    #     # print(pathloss)
    #     self.pathloss = pathloss
    #     return pathloss

    # def pathloss_calculation_dataset_two(self, rsrp2):
    #     # Use to calculate the pathloss = rsrp - ssblockpower
    #     global pathloss
    #     data = []
    #     for i in rsrp2:
    #         pathloss = i - self.sspower
    #         data.append(pathloss)
    #     pathloss = pd.DataFrame(data, columns=['Pathloss'])
    #     # print(pathloss)
    #     self.pathloss = pathloss
    #     return pathloss

    def plot_function(self, image_name):
        # print(pathloss)
        # print(pdcp)
        x = self.pathloss1
        y = self.pdcp1
        x2 = self.pathloss2
        y2 = self.pdcp2
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.scatter(x, y, c='r')
        ax1.scatter( x2, y2, c='g')
        plt.gca().invert_xaxis()
        plt.xlabel("Pathloss (dB)")
        plt.ylabel("Throughput (Mbps)")
        return plt.savefig(image_name)

    def machine_learning(self, linear_regression, poly_regression):
        # feature X = pathloss
        # label Y = pdcp
        pathloss = pd.DataFrame(self.pathloss)
        pdcp = pd.DataFrame(self.pdcp)
        ## 1. Fit a linear regression model
        X = pathloss.to_numpy().reshape(-1, 1)
        y = pdcp.to_numpy()

        y = np.where(np.isnan(y), 0, y)
        X = np.where(np.isnan(X), 0, X)

        regr = LinearRegression()
        regr.fit(X, y)
        ## 2.Predict label values based on features and calculate the training error
        y_pred = regr.predict(X)
        tr_error = mean_absolute_error(y, y_pred)
        print('The training error is: ', tr_error)  # print the training error
        print("w1 = ", regr.coef_)  # print the learnt w1
        print("w0 = ", regr.intercept_)  # print the learnt w0
        ## visualize the model you have learnt, you are supposed to see the datapoints and the fitted h(x), a straignt line

        pathloss2 = pd.DataFrame(self.pathloss2)
        pdcp2 = pd.DataFrame(self.pdcp2)
        ## 1. Fit a linear regression model
        X2 = pathloss2.to_numpy().reshape(-1, 1)
        y2 = pdcp2.to_numpy()

        y2 = np.where(np.isnan(y2), 0, y2)
        X2 = np.where(np.isnan(X2), 0, X2)

        regr2 = LinearRegression()
        regr2.fit(X2, y2)
        ## 2.Predict label values based on features and calculate the training error
        y_pred2 = regr2.predict(X2)
        tr_error2 = mean_absolute_error(y2, y_pred2)
        print('The training error is: ', tr_error2)  # print the training error
        print("w1 = ", regr2.coef_)  # print the learnt w1
        print("w0 = ", regr2.intercept_)  # print the learnt w0
        ## visualize the model you have learnt, you are supposed to see the datapoints and the fitted h(x), a straignt line

        fig = plt.figure(figsize=(8, 6))  # create a new figure with size 8*6

        # create a scatter plot of datapoints
        # each datapoint is depicted by a dot in color 'blue' and size '10'
        plt.scatter(X, y, color='b', s=8, label='datapoints from the OLD')
        plt.scatter(X2, y2, color='g', s=8, label='datapoints from the NEW')
        # plot the predictions obtained by the learnt linear hypothesis using color 'red' and label the curve as "h(x)"
        y_pred = regr.predict(X)  # predict using the linear model
        plt.plot(X, y_pred, color='b', label='OLD')
        y_pred2 = regr2.predict(X2)  # predict using the linear model
        plt.plot(X2, y_pred2, color='g', label='NEW')
        plt.gca().invert_xaxis()
        plt.xlabel('Pathloss (dB)', size=15)  # define label for the horizontal axis
        plt.ylabel('Throughput (Mbps)', size=15)  # define label for the vertical axis
        plt.grid(True)
        plt.title('Linear regression model', size=15)  # define the title of the plot
        plt.legend(loc='best', fontsize=14)  # define the location of the legend

        plt.show()  # display the plot on the screen

        degrees = [3]

        tr_errors = []
        tr_errors2 = []

        fig2 = plt.figure(figsize=(8, 6))
        for i in range(len(degrees)):  # use for-loop to fit polynomial regression models with different degrees
            print("Polynomial degree = ", degrees[i])

            poly = PolynomialFeatures(degree=degrees[i])  # initialize a polynomial feature transformer
            X_poly = poly.fit_transform(X)  # fit and transform the raw features

            lin_regr = LinearRegression(
                fit_intercept=False)  # NOTE: "fit_intercept=False" as we already have a constant iterm in the new feature X_poly
            lin_regr.fit(X_poly, y)  # fit linear regression to these new features and labels (labels remain same)

            y_pred = lin_regr.predict(X_poly)  # predict using the learnt linear model
            tr_error = mean_absolute_error(y, y_pred)  # calculate the training error

            print("The first two row of X_poly: \n", X_poly[0:2])

            print("\nThe learned weights: \n", lin_regr.coef_)

            tr_errors.append(tr_error)
            # X_fit = np.linspace(-70, -102, 350)  # generate samples

            plt.plot(X, lin_regr.predict(poly.transform(X)),
                     label="Model_OLD", c="b")  # plot the polynomial regression model
            plt.scatter(X, y, color="b", s=10,
                        label="datapoints from the OLD_BUILD ")  # plot a scatter plot of y(throughput) vs. X(pathlosss) with color 'blue' and size '10'

            X_poly2 = poly.fit_transform(X2)  # fit and transform the raw features

            lin_regr2 = LinearRegression(
                fit_intercept=False)  # NOTE: "fit_intercept=False" as we already have a constant iterm in the new feature X_poly
            lin_regr2.fit(X_poly2, y2)  # fit linear regression to these new features and labels (labels remain same)

            y_pred2 = lin_regr.predict(X_poly2)  # predict using the learnt linear model
            tr_error = mean_absolute_error(y2, y_pred2)  # calculate the training error

            print("The first two row of X_poly: \n", X_poly2[0:2])

            print("\nThe learned weights: \n", lin_regr2.coef_)

            tr_errors2.append(tr_error)
            # X_fit2 = np.linspace(-70, -102, 350)  # generate samples

            plt.plot(X2, lin_regr2.predict(poly.transform(X2)),
                     label="Model_NEW", c="g")  # plot the polynomial regression model

            plt.scatter(X2, y2, color="g", s=10,
                        label="datapoints from the NEW_BUILD ")

            plt.gca().invert_xaxis()
            plt.xlabel('Pathloss (dB)')  # set the label for the x/y-axis
            plt.ylabel('Throughput (Mbps)')
            plt.grid(True)
            plt.legend(loc="best")  # set the location of the legend
            plt.title('Polynomial degree = {}\nTraining error = {:.5}'.format(degrees[i], tr_error))  # set the title
            plt.show()  # show the plot

        return fig.savefig(linear_regression, dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5), fig2.savefig(
            poly_regression, dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)

# df = pd.read_csv("rsrp_vs_pdcp_DL_mob.csv", sep=';', engine='python')
# df2 = pd.read_csv("rsrp_vs_pdcp_UL_mob.csv", sep=';', engine='python')
# pro = Processor(df)
# pro2 = Processor(df2)

# pdcp_1, pathloss1 = pro.csv_handler()
# pdcp_2, pathloss2 = pro2.csv_handler()

# plott = Plotter()
# plott.plot_function(pathloss1, pdcp_1, pathloss2, pdcp_2)

# pro.ssblock_power = pro.ssBlockPower_input()
# pro.pathloss_calculation(pro.rsrp_, pro.ssblock_power)
# pro.plot_function(pro.pathloss, pro.pdcp_)
# pro.machine_learning(pro.pathloss, pro.pdcp_)
