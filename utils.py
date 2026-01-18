import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import properscoring

##dep
'''def MAE(series, t, forecast_res, steps=None):
    if steps is None:
        steps = len(series) - t

    test = series.iloc[t:t + steps]

    forecast = forecast_res.predicted_mean
    forecast.index = test.index  

    mae = np.mean(np.abs(test - forecast))
    return mae'''


'''
outpus the fraction of trajectories within a given confidence interval accross time
'''
def fractionMetric(trajectories, ci_lower,ci_upper):
    '''
    :trajectories: torch tensor of trajectories (2 dimensional, columns=step, rows=trajectory)
    :ci_lower : confidence interval lower bound
    :ci_upper : confidence interval upper
    :xAxis: indices for plotting 
    '''    
    if not torch.is_tensor(ci_lower):
        ci_lower = torch.tensor(ci_lower.values if hasattr(ci_lower, "values") else ci_lower)

    if not torch.is_tensor(ci_upper):
        ci_upper = torch.tensor(ci_upper.values if hasattr(ci_upper, "values") else ci_upper)
    ci_lower = ci_lower.unsqueeze(0)
    ci_upper = ci_upper.unsqueeze(0) 

    inside = (trajectories >= ci_lower) & (trajectories <= ci_upper)

    fraction = inside.float().mean(dim=0)

    return fraction


def meanMAE(data, sims):
    maes = [mean_absolute_error(data, sim) for sim in sims]    
    mean_mae = np.mean(maes)
    
    return mean_mae

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
    plt.title(f'ARIMA/SARIMAX Forecast, MAE:{MAE(series, t, forecast_res, steps=steps):.4f}, CRPS:{CRPS(result, test_values=test, steps=steps):.4f}')

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

