defmodule StochTree.Forest do
  @moduledoc """
  A trained BART forest (opaque NIF resource + metadata).
  """

  defstruct [:resource, :n_features, :num_trees, :num_samples]

  @type t :: %__MODULE__{
    resource: reference(),
    n_features: non_neg_integer(),
    num_trees: non_neg_integer(),
    num_samples: non_neg_integer()
  }
end
