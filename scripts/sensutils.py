"""Collect useful common functions for the sensitivity analysis."""
import os
import json
import numpy as np
import pandas as pd
from scipy.special import erf


def get_mode_skew(mu, sigma, alpha):
    """Return (wikipedia) numerical approximation of mode."""
    delta = alpha / np.sqrt(1 + alpha**2)
    mu_z = np.sqrt(2 / np.pi) * delta
    sigma_z = np.sqrt(1 - mu_z**2)
    gamma_1 = (4 - np.pi) / 2. * mu_z**3 / (1 - mu_z**2)**1.5
    m0 = mu_z - gamma_1*sigma_z/2. - np.sign(alpha)/2.*np.exp(-2*np.pi/np.abs(alpha))
    return mu + sigma*m0


def logpdf_skewnorm(x, alpha, loc=0, scale=1):
    """Implement skewnorm.logpdf for speed."""
    norm_term = -0.5*((x - loc) / scale)**2
    skew_term = np.log(0.5*(1. + erf(alpha*(x - loc) / (np.sqrt(2)*scale))))
    return np.log(2) - 0.5*np.log(2*np.pi) + norm_term + skew_term


def get_results(name, JSON_FILE):
    """Get a dataframe 'name' from the results file."""
    # Check if the file exists
    if not os.path.exists(JSON_FILE):
        return None

    with open(JSON_FILE, 'r') as file:
        data = json.load(file)

    json_df = data.get(name)
    if json_df is None:
        return None

    # Convert the JSON object back to DataFrame
    return pd.read_json(json_df)


def update_results(name, dataframe, JSON_FILE):
    """Write a dataframe to the results file."""
    data = {}
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as file:
            data = json.load(file)

    # Convert the DataFrame to JSON and update the dictionary
    data[name] = dataframe.to_json()

    # Write the updated dictionary to the JSON file
    with open(JSON_FILE, 'w') as file:
        json.dump(data, file)
