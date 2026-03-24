mod tree;
mod sampler;

use rustler::{Atom, Env, NifResult, ResourceArc, Term};
use std::sync::Mutex;

mod atoms {
    rustler::atoms! {
        ok,
        error,
    }
}

/// Opaque resource holding a trained BART forest
pub struct ForestResource {
    pub forest: Mutex<sampler::BartForest>,
}

fn on_load(env: Env, _info: Term) -> bool {
    rustler::resource!(ForestResource, env);
    true
}

/// Fit a BART model: (x_flat, y, n_obs, n_features, opts) -> {:ok, resource}
#[rustler::nif(schedule = "DirtyCpu")]
fn fit_bart(
    x_flat: Vec<f64>,
    y: Vec<f64>,
    n_obs: usize,
    n_features: usize,
    num_trees: usize,
    num_gfr: usize,
    num_mcmc: usize,
    seed: u64,
) -> NifResult<(Atom, ResourceArc<ForestResource>)> {
    let forest = sampler::fit_bart(
        &x_flat, &y, n_obs, n_features,
        num_trees, num_gfr, num_mcmc, seed,
    );

    let resource = ResourceArc::new(ForestResource {
        forest: Mutex::new(forest),
    });

    Ok((atoms::ok(), resource))
}

/// Predict: (resource, x_new_flat, n_new, n_features) -> posterior_samples (flat)
#[rustler::nif(schedule = "DirtyCpu")]
fn predict(
    resource: ResourceArc<ForestResource>,
    x_flat: Vec<f64>,
    n_obs: usize,
    n_features: usize,
) -> NifResult<Vec<Vec<f64>>> {
    let forest = resource.forest.lock().unwrap();
    let predictions = forest.predict(&x_flat, n_obs, n_features);
    Ok(predictions)
}

/// Variable importance: split counts per feature
#[rustler::nif]
fn variable_importance(
    resource: ResourceArc<ForestResource>,
) -> NifResult<Vec<(usize, usize)>> {
    let forest = resource.forest.lock().unwrap();
    let importance = forest.variable_importance();
    Ok(importance)
}

/// Get sigma samples from the fitted model
#[rustler::nif]
fn sigma_samples(
    resource: ResourceArc<ForestResource>,
) -> NifResult<Vec<f64>> {
    let forest = resource.forest.lock().unwrap();
    Ok(forest.sigma_samples.clone())
}

rustler::init!("Elixir.StochTree.Native", [
    fit_bart,
    predict,
    variable_importance,
    sigma_samples,
], load = on_load);
