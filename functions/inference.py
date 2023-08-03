import arviz as az
import jax.numpy as jnp

################################################################################
################################################################################
##############################   MAIN FUNCTIONS   ##############################
################################################################################
################################################################################

import arviz as az
import jax.numpy as jnp

def calc_mean_hpdi(arviz_post, ci=0.89, y_scaler=None, mu_var='mu', sim_var='obs'):
    """
    Calculate the mean and highest posterior density interval (HPDI) for the 'mu' and 'obs' variables in the ArviZ posterior
    and posterior predictive objects.

    Parameters
    ----------
    arviz_post : ArviZ InferenceData object
        The posterior and posterior predictive samples.
    ci : float, optional
        The probability mass of the HPDI. Default is 0.89.
    y_scaler : Scaler object, optional
        A Scaler object to unscale the mean and HPDI values if the data was scaled before fitting the model. Default is None.
    mu_var : str, optional
        The name of the 'mu' variable in the ArviZ posterior object. Default is 'mu'.
    sim_var : str, optional
        The name of the 'obs' variable in the ArviZ posterior predictive object. Default is 'obs'.

    Returns
    -------
    mean_mu : ndarray
        The mean of the 'mu' variable.
    hpdi_mu : ndarray
        The HPDI of the 'mu' variable.
    hpdi_sim : ndarray
        The HPDI of the 'obs' variable.
    """
    # Define the dimensions that are common to both the posterior and posterior predictive objects
    base_dims = ['chain','draw']
    
    # Get the mean of the 'mu' variable in the posterior object
    mean_mu = arviz_post.posterior[mu_var].mean(dim=base_dims).values
    
    # Define the dimensions for the 'mu' and 'obs' variables that are not in the base dimensions
    mu_dims = [_ for _ in arviz_post.posterior[mu_var].coords.dims if not _ in base_dims]
    sim_dims = [_ for _ in arviz_post.posterior_predictive[sim_var].coords.dims if not _ in base_dims]
    
    # Calculate the HPDI for the 'mu' and 'obs' variables using the arviz.hdi() function
    hpdi_mu = az.hdi(
        arviz_post.posterior, hdi_prob=ci, var_names=[mu_var]
    ).transpose('hdi',*mu_dims)[mu_var].values
    hpdi_sim = az.hdi(
        arviz_post.posterior_predictive, hdi_prob=ci, var_names=[sim_var]
    ).transpose('hdi',*sim_dims)[sim_var].values

    # If a scaler object is provided, unscale the mean and HPDI values and reverse the log transform if necessary
    if not y_scaler is None:
        # Unscale the mean and HPDI values
        mean_mu = y_scaler.inverse_transform(mean_mu)
        hpdi_mu[0,...] = y_scaler.inverse_transform(hpdi_mu[0,...])
        hpdi_mu[1,...] = y_scaler.inverse_transform(hpdi_mu[1,...])
        hpdi_sim[0,...] = y_scaler.inverse_transform(hpdi_sim[0,...])
        hpdi_sim[1,...] = y_scaler.inverse_transform(hpdi_sim[1,...])
        # Reverse the log transform
        mean_mu = jnp.exp(mean_mu)
        hpdi_mu = jnp.exp(hpdi_mu)
        hpdi_sim = jnp.exp(hpdi_sim)
    
    # Return the mean and HPDI values for the 'mu' and 'obs' variables
    return mean_mu, hpdi_mu, hpdi_sim
        
################################################################################
################################################################################
