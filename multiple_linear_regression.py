import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

'''
multiple linear regression
get input cleaned dataframe and dataframe for regression
'''

def multiple_linear_regression(df,df_for_reg):
    X = df_for_reg
    y = df['latest_price']

    # Multiple Linear Regression with kfold k=5 , k= 10
    for i in range(5, 11, 5):
        print('\n=== kfold k is {} ==='.format(i))
        Linear_model = LinearRegression()
        kf = KFold(n_splits=i, shuffle=True, random_state=50)
        accuracy_history = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            Linear_model.fit(X_train, y_train)
            y_pred = Linear_model.predict(X_test)
            accuracy_history.append(Linear_model.score(X_test, y_test))

        # Evaluation with R squared
        print("each fold R square :", accuracy_history)
        print("mean R square :", np.mean(accuracy_history))
        print("coef: ", Linear_model.coef_)
        print("intercept: ", Linear_model.intercept_)

        # visualize result
        plt.scatter(y_test, y_pred, alpha=0.4, c='red')
        plt.xlabel("Latest price ")
        plt.ylabel("Predicted price")
        plt.title("MULTIPLE LINEAR REGRESSIONkfold {}".format(i))
        plt.show()

        plt.plot(Linear_model.predict(X_test), label="predicted price")
        plt.plot(y_test.values.reshape(-1, 1), label="latest price ")
        plt.title("MULTIPLE LINEAR REGRESSION kfold {}".format(i))
        plt.legend()
        plt.show()

        # coefficients visulization
        plt.figure(figsize=(10, 8))
        plt.barh(['Processor_name', 'Processor_gnrtn', 'Ram_gb', 'Ram_type', 'SSD', 'HDD', 'MAC', 'WINDOWS', 'OS_bit',
                  'Graphic_card_gb'
                     , 'Weight', 'Display_size', 'Warranty', 'Touchscreen', 'Msoffice', 'Old_price', 'Discount',
                  'Star_rating', 'Ratings', 'Reviews'], np.ravel(Linear_model.coef_))

        plt.title('Coefficient of feature kfold {}'.format(i), fontsize=20)
        plt.xlabel('Coefficient', fontsize=18)
        plt.show()

        #Multiple Linear Regression with Holdout method
        #train:test | 90:10 | 80:20 | 70:30
        for i in range(1, 4):
            print('\n=== train{} | test{} ==='.format((10 - i) / 10, i / 10))

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=i / 10, random_state=1, shuffle=True)

            # get model
            linear_reg = LinearRegression()

            # fit the model to the training data
            linear_reg.fit(x_train, y_train)

            y_pred = linear_reg.predict(x_test)

            # Evaluation with R squared
            print("R squared :", linear_reg.score(x_test, y_test))
            print('coef: {}'.format(linear_reg.coef_))
            print('intercept: {}'.format(linear_reg.intercept_))

            # visualize result
            plt.scatter(y_test, y_pred, alpha=0.4, c='red')
            plt.xlabel("Latest price ")
            plt.ylabel("Predicted price")
            plt.title("MULTIPLE LINEAR REGRESSION train{} test{}".format((10 - i) / 10, i / 10))
            plt.show()

            plt.plot(linear_reg.predict(x_test), label="predicted price")
            plt.plot(y_test.values.reshape(-1, 1), label="latest price ")
            plt.title("MULTIPLE LINEAR REGRESSION train{} test{}".format((10 - i) / 10, i / 10))
            plt.legend()
            plt.show()

            # coefficients visulization
            plt.figure(figsize=(10, 8))
            plt.barh(
                ['Processor_name', 'Processor_gnrtn', 'Ram_gb', 'Ram_type', 'SSD', 'HDD', 'MAC', 'WINDOWS', 'OS_bit',
                 'Graphic_card_gb', 'Weight', 'Display_size', 'Warranty', 'Touchscreen', 'Msoffice', 'Old_price',
                 'Discount',
                 'Star_rating', 'Ratings', 'Reviews'], np.ravel(linear_reg.coef_))
            plt.title('Coefficient of feature train{} test{}'.format((10 - i) / 10, i / 10), fontsize=20)
            plt.xlabel('Coefficient', fontsize=18)

            plt.show()