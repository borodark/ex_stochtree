# Head-to-head: StochTree-Ex vs StochTree Python 0.4.0
# Runs matching BART configs and reports wall time + RMSE.
#
# Usage:
#   cd stochtree_ex && elixir --erl "+sbt tnnps" -S mix run benchmark/head_to_head.exs
#
# Compare with:
#   ~/projects/learn_erl/python-env/bin/python benchmark/head_to_head.py

defmodule HeadToHead do
  def friedman1(n, p_noise \\ 5, seed \\ 42) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})
    p = 5 + p_noise
    x = for _ <- 1..n, do: for(_ <- 1..p, do: :rand.uniform())
    y = Enum.map(x, fn row ->
      [x1, x2, x3, x4, x5 | _] = row
      10 * :math.sin(:math.pi() * x1 * x2) +
        20 * :math.pow(x3 - 0.5, 2) +
        10 * x4 + 5 * x5 + :rand.normal()
    end)
    {x, y}
  end

  def friedman2(n, seed \\ 42) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})
    x = for _ <- 1..n do
      [
        :rand.uniform() * 100,
        :rand.uniform() * (560 - 40) * :math.pi() + 40 * :math.pi(),
        :rand.uniform(),
        :rand.uniform() * 10 + 1
      ]
    end
    y = Enum.map(x, fn [x1, x2, x3, x4] ->
      val = :math.sqrt(x1 * x1 + :math.pow(x2 * x3 - 1 / (x2 * x4), 2))
      val + :rand.normal() * val * 0.1
    end)
    {x, y}
  end

  def simple_linear(n, seed \\ 42) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})
    x = for _ <- 1..n, do: for(_ <- 1..3, do: :rand.uniform() * 10)
    y = Enum.map(x, fn [x1 | _] -> 3.0 * x1 + :rand.normal() * 0.5 end)
    {x, y}
  end

  def high_p(n, p, seed \\ 42) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})
    x = for _ <- 1..n, do: for(_ <- 1..p, do: :rand.uniform())
    y = Enum.map(x, fn row -> Enum.at(row, 0) * 2.0 + :rand.normal() * 0.5 end)
    {x, y}
  end

  def rmse(y_true, y_pred) do
    n = length(y_true)
    ss = Enum.zip(y_true, y_pred)
         |> Enum.map(fn {a, b} -> (a - b) * (a - b) end)
         |> Enum.sum()
    :math.sqrt(ss / n)
  end

  def run_test(name, x, y, opts) do
    t0 = System.monotonic_time(:millisecond)
    {forest, _sigmas} = StochTree.BART.fit(x, y, opts)
    elapsed = System.monotonic_time(:millisecond) - t0

    %{mean: preds} = StochTree.predict(forest, x)
    r = rmse(y, preds)
    {name, elapsed, Float.round(r, 4)}
  end
end

IO.puts(String.duplicate("=", 70))
IO.puts("StochTree-Ex (Elixir/Rust) — Head-to-Head")
IO.puts("#{System.schedulers_online()} schedulers")
IO.puts(String.duplicate("=", 70))
IO.puts("")

tests = [
  {"smoke (n=1K, p=10)", fn -> HeadToHead.friedman1(1000, 5) end,
   [num_trees: 200, num_gfr: 100, num_mcmc: 100, seed: 42]},
  {"friedman2 (n=500, p=4)", fn -> HeadToHead.friedman2(500) end,
   [num_trees: 200, num_gfr: 10, num_mcmc: 100, seed: 42]},
  {"simple-linear (n=200, p=3)", fn -> HeadToHead.simple_linear(200) end,
   [num_trees: 50, num_gfr: 5, num_mcmc: 50, seed: 42]},
  {"high-p-100 (n=1K, p=100)", fn -> HeadToHead.high_p(1000, 100) end,
   [num_trees: 200, num_gfr: 100, num_mcmc: 100, seed: 42]},
  {"extreme-p-500 (n=1K, p=500)", fn -> HeadToHead.high_p(1000, 500) end,
   [num_trees: 200, num_gfr: 100, num_mcmc: 100, seed: 42]},
  {"medium-n (n=5K, p=10)", fn -> HeadToHead.friedman1(5000, 5) end,
   [num_trees: 200, num_gfr: 100, num_mcmc: 100, seed: 42]},
  {"large-n (n=10K, p=10)", fn -> HeadToHead.friedman1(10000, 5) end,
   [num_trees: 200, num_gfr: 100, num_mcmc: 100, seed: 42]},
]

header = String.pad_trailing("Test", 35) <> String.pad_leading("Time", 10) <> String.pad_leading("RMSE", 10)
IO.puts(header)
IO.puts(String.duplicate("-", 60))

results = Enum.map(tests, fn {name, gen_fn, opts} ->
  {x, y} = gen_fn.()
  {name, elapsed, rmse} = HeadToHead.run_test(name, x, y, opts)
  line = "  " <> String.pad_trailing(name, 33) <>
    String.pad_leading("#{elapsed}ms", 8) <>
    String.pad_leading("#{rmse}", 10)
  IO.puts(line)
  %{name: name, time_ms: elapsed, rmse: rmse}
end)

IO.puts("")
IO.puts(String.duplicate("=", 70))

json = Jason.encode!(results, pretty: true)
File.write!("benchmark/elixir_head_to_head.json", json)
IO.puts("Saved to benchmark/elixir_head_to_head.json")
