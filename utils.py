import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import properscoring

def MAE(series, t, forecast_res, steps=None):
    if steps is None:
        steps = len(series) - t

    test = series.iloc[t:t + steps]

    forecast = forecast_res.predicted_mean
    forecast.index = test.index  

    mae = np.mean(np.abs(test - forecast))
    return mae

def CRPS(result, test_values, n_sim=100, steps=None):
    if steps is None:
        steps = len(test_values)
    y_true = np.array(test_values)[:steps]
    sims = []
    for _ in range(n_sim):
        sim = result.simulate(nsimulations=steps, anchor="end")
        sim_arr = np.array(sim)  # convert Series/DataFrame to np array
        sim_arr = sim_arr.ravel() # flatten to 1D
        sims.append(sim_arr)
    future_samples = np.stack(sims, axis=0)  # shape (n_sim, steps)
    return np.mean(properscoring.crps_ensemble(y_true, forecasts=future_samples, axis=0))

def plot_arima_simulated_forecast(series, result, t, steps=None, n_sim=1,plot_mean=False, show_ground_truth=True, alpha=0.05):
    if steps is None:
        steps = len(series) - t

    #train = series.iloc[:t]
    test = series.iloc[t:t+steps] #how many into the future to forecast

    forecast_res = result.get_forecast(steps=steps)
    ci = forecast_res.conf_int(alpha=alpha)
    ci.index = test.index

    #last_value = series.iloc[t-1]

    sims = []

    for _ in range(n_sim):
        sim = result.simulate(
            nsimulations=steps,
            anchor="end"
        )
        sim.index = test.index
        sims.append(sim)



    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series, color="black", label="Full series")

    if show_ground_truth:
        plt.plot(test.index, test, color="blue", label="Ground truth")

    if plot_mean:
        forecast = forecast_res.predicted_mean
        forecast.index = test.index  
        plt.plot(forecast.index, forecast, color="red", label="ARIMA/SARIMAX forecast mean")

    for sim in sims:
        plt.plot(sim.index, sim, color="orange", alpha=0.5, label="Simulated path" if n_sim == 1 else None)

    plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="red", alpha=0.2, label=f'{100*(1-alpha)}% Confidence interval')

    plt.xlabel("Time (steps)")
    plt.ylabel("Value")
    plt.title(f'ARIMA/SARIMAX Forecast Overlay on Full Series, MAE:{MAE(series, t, forecast_res, steps=steps):.4f}, CRPS:{CRPS(result, test_values=test, steps=steps):.4f}')

    plt.legend()
    plt.tight_layout()
    plt.show()


#################################
#################################
#################################
################################# deprecated below

def plot_arima_forecast_mean(series, result, t, steps=None, conf_int=True):
    """
    Plot ARIMA forecast vs ground truth, overlaying forecast mean and CI.

    Parameters
    ----------
    series : pd.Series
        Full time series (ground truth)
    result : ARIMAResults
        Fitted ARIMA model result
    t : int
        Time index where forecasting starts
    steps : int, optional
        Number of forecast steps (default: len(series) - t)
    conf_int : bool
        Plot confidence interval
    """
    if steps is None:
        steps = len(series) - t

    train = series.iloc[:t+1]
    test = series.iloc[t:t + steps]

    forecast_res = result.get_forecast(steps=steps)
    forecast = forecast_res.predicted_mean
    ci = forecast_res.conf_int()

    forecast.index = test.index  
    ci.index = test.index

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Training data", color="black")
    plt.plot(test.index, test, label="Ground truth", color="blue")
    plt.plot(forecast.index, forecast, label="ARIMA forecast mean", color="red")

    if conf_int:
        plt.fill_between(
            ci.index,
            ci.iloc[:, 0],
            ci.iloc[:, 1],
            color="red",
            alpha=0.2,
            label="Confidence interval"
        )

    plt.axvline(train.index[-1], color="gray", linestyle="--", label="Forecast start")

    plt.legend()
    plt.title("ARIMA Forecast vs Ground Truth")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()



def convert_to_series(array):
    """
    Convert a numpy array or torch tensor to a Pandas Series.

    Parameters:
    array (np.ndarray or torch.Tensor): Input NumPy array or torch tensor.
    or 
    array (list): Input list.
    or array (torch.Tensor): Input torch tensor.
    Returns:
    pd.Series: Converted Pandas Series.
    """
    if isinstance(array, torch.Tensor):
        array = array.numpy()
    elif isinstance(array, list):
        array = np.array(array)
    return pd.Series(array)

