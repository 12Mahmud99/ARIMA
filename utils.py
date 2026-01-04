import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels 
import torch

import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

def plot_arima_simulated_forecast(series, result, t, steps=None, n_sim=1, show_ground_truth=True):
    """
    Plot ARIMA forecast with simulated paths + confidence intervals.

    Parameters
    ----------
    series : pd.Series
        Full time series (for indexing and optional ground truth)
    result : ARIMAResults
        Fitted ARIMA model
    t : int
        Time index where forecast starts
    steps : int, optional
        Forecast horizon (default: remaining length)
    n_sim : int
        Number of simulated paths to plot
    show_ground_truth : bool
        Overlay the actual future values
    """
    if steps is None:
        steps = len(series) - t

    train = series.iloc[:t]
    test = series.iloc[t:t+steps]

    forecast_res = result.get_forecast(steps=steps)
    #forecast_mean = forecast_res.predicted_mean
    ci = forecast_res.conf_int()
    #forecast.index = test.index  
    ci.index = test.index

    sims = []
    for _ in range(n_sim):
        sim = result.simulate(nsimulations=steps)
        sim.index = test.index
        sims.append(sim)

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, color="black", label="Training data")
    if show_ground_truth:
        plt.plot(test.index, test, color="blue", label="Ground truth")

    for sim in sims:
        plt.plot(sim.index, sim, color="orange", alpha=0.5, label="Simulated path" if n_sim == 1 else None)

    #plt.plot(forecast_mean.index, forecast_mean, color="red", label="Forecast mean")
    plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="red", alpha=0.2, label="Confidence interval")

    plt.axvline(train.index[-1], color="gray", linestyle="--", label="Forecast start")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("ARIMA Forecast with Simulated Paths + CI")
    plt.legend()
    plt.tight_layout()
    plt.show()

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

    train = series.iloc[:t]
    test = series.iloc[t:t + steps]

    forecast_res = result.get_forecast(steps=steps)
    forecast = forecast_res.predicted_mean
    ci = forecast_res.conf_int()

    forecast.index = test.index  
    ci.index = test.index

    plt.figure(figsize=(12, 6))

    # Plot
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

