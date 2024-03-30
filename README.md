## Project Title: 
Conditional Tabular Generative Adversarial Network Based Temperature Forecasting System

## Introduction:
This project utilizes Temporal Convolutional Networks (TCN) and Conditional Tabular Generative Adversarial Network (CTGAN) to forecast temperature. The research findings have been published in IEEE Transactions on Sustainable Computing under the paper titled "Conditional Tabular Generative Adversarial Network Based Temperature Forecasting System". The dataset used for this project can be found at [Kaggle](https://www.kaggle.com/datasets/ananthr1/weather-prediction).

## Files Description:
1. **data:** 
   - `seattle-weather.csv`: Original dataset from Kaggle. Folders `temp_max` and `temp_min` are utilized for storing training, testing, and synthetic numpy data.
   - `data_plot`: Numpy data from `model.py` used for plotting figures.

2. **figure:** 
   - Directory to save figures of prediction outcomes.

3. **data_gan.py:** 
   - Script to produce training, testing, and synthetic numpy data.

4. **model.py:** 
   - Contains three methods for building the model: TCN, TCN+CTGAN, TCN+CTGAN(processed).

5. **plot.py:** 
   - Script to plot prediction, residuals, and algorithms comparison figures.

6. **predict.py:** 
   - Function to plot prediction and residuals figures.

## How to Use:
1. Execute `data_gan.py`.
2. Execute `model.py` (If you already have all numpy data, you can directly execute `model.py`).
3. Ensure not to change the path of each file.

Note: Ensure the proper execution order mentioned above for seamless functioning of the project.
