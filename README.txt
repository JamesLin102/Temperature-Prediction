This project is about using TCN and CTGAN to predict temperature.
The research is published on IEEE Transactions on Sustainable Computing.
The paper title is Conditional Tabular Generative Adversarial Network Based Temperature Forecasting System.
Dataset is from https://www.kaggle.com/datasets/ananthr1/weather-prediction
---------------------------------------------------------------------------------
Files description
1. data: seattle-weather.csv is original dataset from Kaggle. temp_max&min are uesd to save traning, testing and synthetic numpy data. data_plot is also numpy data from modle.py to plot the figure.
2. figure: To save figure of prediction outcome.
3. data_gan.py: To producce traning, testing and synthetic numpy data.
4. model.py: There are three way to built the model. (TCN, TCN+CTGAN, TCN+CTGAN(processed))
5. plot.py: To plot prediction, residuals and algorithms comparison figure.
6. predict.py: Function of ploting prediction and residuals figure.
---------------------------------------------------------------------------------
How to use?
data_gan.py -> modle.py (If you have alreaedy gotten all numpy data, you can only execute modle.py.)
!! Don't change the path of each files !!