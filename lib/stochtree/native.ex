defmodule StochTree.Native do
  @moduledoc false
  use Rustler, otp_app: :stochtree_ex, crate: "stochtree_nif"

  def fit_bart(_x_flat, _y, _n_obs, _n_features, _num_trees, _num_gfr, _num_mcmc, _seed),
    do: :erlang.nif_error(:nif_not_loaded)

  def predict(_resource, _x_flat, _n_obs, _n_features),
    do: :erlang.nif_error(:nif_not_loaded)

  def variable_importance(_resource),
    do: :erlang.nif_error(:nif_not_loaded)

  def sigma_samples(_resource),
    do: :erlang.nif_error(:nif_not_loaded)
end
