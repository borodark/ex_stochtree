defmodule StochTree.MixProject do
  use Mix.Project

  def project do
    [
      app: :stochtree_ex,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "Bayesian Additive Regression Trees (BART) for Elixir via StochTree C++",
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:rustler, "~> 0.36", runtime: false},
      {:nx, "~> 0.10", optional: true},
      {:jason, "~> 1.4"}
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/borodark/stochtree_ex"}
    ]
  end
end
