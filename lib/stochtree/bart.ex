defmodule StochTree.BART do
  @moduledoc """
  Bayesian Additive Regression Trees.

  Nonparametric Bayesian regression: `y = f(X) + noise` where `f` is
  a sum of many small decision trees. No functional form specified —
  the model discovers nonlinearities and interactions automatically.

  ## Example

      {forest, sigmas} = StochTree.BART.fit(x_train, y_train,
        num_trees: 200,
        num_gfr: 10,
        num_mcmc: 100
      )

      predictions = StochTree.predict(forest, x_test)

  ## Options

  - `:num_trees` — number of trees in the ensemble (default: 200)
  - `:num_gfr` — GFR (grow-from-root) warm-start iterations (default: 10)
  - `:num_mcmc` — MCMC iterations after GFR (default: 100)
  - `:seed` — random seed (default: 42)
  """

  alias StochTree.Native

  @default_opts [
    num_trees: 200,
    num_gfr: 10,
    num_mcmc: 100,
    seed: 42
  ]

  @doc """
  Fit a BART model to training data.

  ## Arguments

  - `x` — feature matrix, one of:
    - List of lists: `[[1.0, 2.0], [3.0, 4.0]]` (row-major)
    - Nx tensor of shape `{n, p}`
    - Flat list with `n_features` option
  - `y` — response vector (list of floats)
  - `opts` — keyword options (see module docs)

  ## Returns

  `{%StochTree.Forest{}, sigma_samples}` where sigma_samples is a list
  of posterior draws of the observation noise standard deviation.
  """
  def fit(x, y, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)
    {x_flat, n_obs, n_features} = flatten_x(x, opts)
    y_flat = to_float_list(y)

    if length(y_flat) != n_obs do
      raise ArgumentError, "y length #{length(y_flat)} != n_obs #{n_obs}"
    end

    {:ok, resource} = Native.fit_bart(
      x_flat, y_flat, n_obs, n_features,
      opts[:num_trees], opts[:num_gfr], opts[:num_mcmc], opts[:seed]
    )

    forest = %StochTree.Forest{
      resource: resource,
      n_features: n_features,
      num_trees: opts[:num_trees],
      num_samples: opts[:num_gfr] + opts[:num_mcmc]
    }

    sigma_samples = Native.sigma_samples(resource)

    {forest, sigma_samples}
  end

  # --- Data conversion ---

  defp flatten_x(x, opts) when is_list(x) do
    cond do
      # List of lists (row-major)
      is_list(hd(x)) ->
        n_obs = length(x)
        n_features = length(hd(x))
        x_flat = List.flatten(x) |> Enum.map(&to_float/1)
        {x_flat, n_obs, n_features}

      # Flat list with explicit n_features
      is_number(hd(x)) and Keyword.has_key?(opts, :n_features) ->
        n_features = opts[:n_features]
        n_obs = div(length(x), n_features)
        x_flat = Enum.map(x, &to_float/1)
        {x_flat, n_obs, n_features}

      true ->
        raise ArgumentError, "x must be list-of-lists or flat list with :n_features option"
    end
  end

  # Nx tensor support (optional)
  defp flatten_x(x, _opts) do
    if Code.ensure_loaded?(Nx) do
      shape = Nx.shape(x)
      case tuple_size(shape) do
        2 ->
          {n_obs, n_features} = shape
          x_flat = Nx.to_flat_list(x)
          {x_flat, n_obs, n_features}
        _ ->
          raise ArgumentError, "Nx tensor must be 2D {n_obs, n_features}"
      end
    else
      raise ArgumentError, "x must be list-of-lists, flat list, or Nx tensor"
    end
  end

  defp to_float_list(y) when is_list(y), do: Enum.map(y, &to_float/1)
  defp to_float_list(y) do
    if Code.ensure_loaded?(Nx), do: Nx.to_flat_list(y), else: raise("y must be a list")
  end

  defp to_float(x) when is_float(x), do: x
  defp to_float(x) when is_integer(x), do: x * 1.0
end
