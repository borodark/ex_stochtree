defmodule StochTree do
  @moduledoc """
  Bayesian Additive Regression Trees (BART) for Elixir.

  Nonparametric Bayesian regression and causal inference via tree ensembles.
  Wraps the BART algorithm with a Rust NIF for performance.

  ## Quick Start

      # Fit
      {forest, sigmas} = StochTree.BART.fit(x_train, y_train,
        num_trees: 200, num_mcmc: 100)

      # Predict with uncertainty
      preds = StochTree.predict(forest, x_test)

      # Feature importance
      importance = StochTree.variable_importance(forest)

  ## When to Use BART vs NUTS/HMC

  - **BART**: unknown functional form, many features, prediction + uncertainty
  - **NUTS**: known parametric model, few parameters, precise posterior inference
  """

  alias StochTree.{Forest, Native}

  @doc """
  Predict for new observations using a trained forest.

  Returns `%{mean: [...], samples: [[...], ...], lower: [...], upper: [...]}`.
  """
  def predict(%Forest{} = forest, x_new, opts \\ []) do
    {x_flat, n_obs, n_features} = flatten_x(x_new, forest.n_features)
    quantiles = Keyword.get(opts, :quantiles, {0.05, 0.95})

    if n_features != forest.n_features do
      raise ArgumentError,
        "x_new has #{n_features} features but forest was trained with #{forest.n_features}"
    end

    samples = Native.predict(forest.resource, x_flat, n_obs, n_features)
    n_samples = length(samples)

    mean = for i <- 0..(n_obs - 1) do
      vals = Enum.map(samples, fn s -> Enum.at(s, i) end)
      Enum.sum(vals) / n_samples
    end

    {q_lo, q_hi} = quantiles

    lower = for i <- 0..(n_obs - 1) do
      vals = Enum.map(samples, fn s -> Enum.at(s, i) end) |> Enum.sort()
      Enum.at(vals, round(n_samples * q_lo))
    end

    upper = for i <- 0..(n_obs - 1) do
      vals = Enum.map(samples, fn s -> Enum.at(s, i) end) |> Enum.sort()
      Enum.at(vals, round(n_samples * q_hi))
    end

    %{mean: mean, lower: lower, upper: upper, samples: samples}
  end

  @doc """
  Variable importance: split counts per feature, sorted by importance.

  Returns `[{feature_index, split_count}, ...]` sorted descending.
  """
  def variable_importance(%Forest{} = forest) do
    Native.variable_importance(forest.resource)
  end

  defp flatten_x(x, expected_features) when is_list(x) do
    cond do
      is_list(hd(x)) ->
        n_obs = length(x)
        n_features = length(hd(x))
        {List.flatten(x) |> Enum.map(&to_f/1), n_obs, n_features}

      is_number(hd(x)) ->
        n_obs = div(length(x), expected_features)
        {Enum.map(x, &to_f/1), n_obs, expected_features}

      true ->
        raise ArgumentError, "invalid x format"
    end
  end

  defp flatten_x(x, _expected) do
    if Code.ensure_loaded?(Nx) do
      {n_obs, n_features} = Nx.shape(x)
      {Nx.to_flat_list(x), n_obs, n_features}
    else
      raise ArgumentError, "x must be list-of-lists or Nx tensor"
    end
  end

  defp to_f(x) when is_float(x), do: x
  defp to_f(x) when is_integer(x), do: x * 1.0
end
