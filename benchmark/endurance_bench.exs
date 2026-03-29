# StochTree-Ex Endurance Benchmark
#
# Stress-tests BART at scale: large n, high p, sparse features,
# smooth functions, real data, multi-chain parallelism.
#
# Usage:
#   mix run benchmark/endurance_bench.exs              # all tests
#   mix run benchmark/endurance_bench.exs --quick       # smoke only
#   mix run benchmark/endurance_bench.exs --test medium  # specific test
#
# Reference: dbarts R package timings on comparable hardware.

quick_mode = "--quick" in System.argv()
target_test = Enum.find_value(System.argv(), fn
  "--test" -> nil
  arg -> if String.starts_with?(arg, "--test="), do: String.replace(arg, "--test=", "")
end)

IO.puts("=" |> String.duplicate(70))
IO.puts("  StochTree-Ex Endurance Benchmark")
IO.puts("  #{System.schedulers_online()} schedulers, bind=#{:erlang.system_info(:scheduler_bind_type)}")
IO.puts("  mode: #{if quick_mode, do: "quick", else: "full"}")
IO.puts("=" |> String.duplicate(70))
IO.puts("")

defmodule EnduranceBench do
  def run_test(name, x, y, opts, expected) do
    IO.write("  #{String.pad_trailing(name, 25)}")
    n = length(y)
    p = length(hd(x))

    {us, {forest, sigmas}} = :timer.tc(fn ->
      StochTree.BART.fit(x, y, opts)
    end)

    ms = div(us, 1000)
    mean_sigma = Enum.sum(sigmas) / max(length(sigmas), 1)

    # Predict on subset
    x_test = Enum.take(x, min(100, n))
    preds = StochTree.predict(forest, x_test)
    y_test = Enum.take(y, min(100, n))

    rmse = preds.mean
      |> Enum.zip(y_test)
      |> Enum.map(fn {p, t} -> (p - t) * (p - t) end)
      |> Enum.sum()
      |> Kernel./(length(y_test))
      |> :math.sqrt()

    # Variable importance
    importance = StochTree.variable_importance(forest)
    top_feat = if importance != [], do: elem(hd(importance), 0), else: -1

    status = if ms < expected * 2, do: "OK", else: "SLOW"

    IO.puts("#{ms}ms  RMSE=#{Float.round(rmse, 3)}  sigma=#{Float.round(mean_sigma, 4)}  top_feat=#{top_feat}  [#{status}]")

    %{name: name, n: n, p: p, ms: ms, rmse: rmse, sigma: mean_sigma, top_feat: top_feat}
  end

  # --- Data generators ---

  def friedman1(n, p_noise \\ 5) do
    :rand.seed(:exsss, {42, 43, 44})
    p = 5 + p_noise
    x = for _ <- 1..n, do: Enum.map(1..p, fn _ -> :rand.uniform() end)
    y = Enum.map(x, fn row ->
      [x1, x2, x3, x4, x5 | _] = row
      10 * :math.sin(:math.pi * x1 * x2) + 20 * (x3 - 0.5) * (x3 - 0.5) +
        10 * x4 + 5 * x5 + :rand.normal()
    end)
    {x, y}
  end

  def smooth_sin(n) do
    :rand.seed(:exsss, {50, 51, 52})
    x = for _ <- 1..n, do: Enum.map(1..5, fn _ -> :rand.uniform() end)
    y = Enum.map(x, fn [x1, x2, x3, x4, x5] ->
      :math.sin(2 * :math.pi * x1) + :math.cos(2 * :math.pi * x2) +
        2 * x3 * x3 + x4 + :rand.normal() * 0.2
    end)
    {x, y}
  end

  def sparse_linear(n, p) do
    :rand.seed(:exsss, {60, 61, 62})
    x = for _ <- 1..n, do: Enum.map(1..p, fn _ -> :rand.uniform() * 10 end)
    y = Enum.map(x, fn row ->
      3.0 * Enum.at(row, 0) + 2.0 * Enum.at(row, 1) - 1.5 * Enum.at(row, 2) +
        0.8 * Enum.at(row, 3) + 0.3 * Enum.at(row, 4) + :rand.normal() * 0.5
    end)
    {x, y}
  end

  def california_housing do
    # Synthetic approximation of California Housing (n=20K, p=8)
    :rand.seed(:exsss, {70, 71, 72})
    n = 20000
    x = for _ <- 1..n, do: [
      :rand.uniform() * 50 + 1,     # MedInc
      :rand.uniform() * 50 + 1,     # HouseAge
      :rand.uniform() * 10 + 1,     # AveRooms
      :rand.uniform() * 5 + 1,      # AveBedrms
      :rand.uniform() * 35000 + 1,  # Population
      :rand.uniform() * 6 + 1,      # AveOccup
      :rand.uniform() * 5 + 32,     # Latitude
      :rand.uniform() * 10 - 124    # Longitude
    ]
    y = Enum.map(x, fn [inc, age, rooms, _, pop, occup, lat, lon] ->
      # Rough approximation of real housing price model
      log_price = 0.4 * :math.log(inc + 1) - 0.1 * :math.log(pop + 1) +
        0.2 * :math.log(rooms + 1) - 0.05 * abs(lat - 37) - 0.03 * abs(lon + 120) +
        0.01 * age - 0.1 * occup + :rand.normal() * 0.3
      :math.exp(log_price) / 100
    end)
    {x, y}
  end
end

# --- Define test suite ---

tests = [
  {"smoke (1K×10)", fn ->
    {x, y} = EnduranceBench.friedman1(1000, 5)
    EnduranceBench.run_test("smoke", x, y,
      [num_trees: 200, num_gfr: 10, num_mcmc: 100, seed: 42], 5_000)
  end},

  {"medium-n (10K×10)", fn ->
    {x, y} = EnduranceBench.friedman1(10_000, 5)
    EnduranceBench.run_test("medium-n", x, y,
      [num_trees: 200, num_gfr: 10, num_mcmc: 100, seed: 42], 60_000)
  end},

  {"large-n (50K×10)", fn ->
    {x, y} = EnduranceBench.friedman1(50_000, 5)
    EnduranceBench.run_test("large-n", x, y,
      [num_trees: 100, num_gfr: 5, num_mcmc: 50, seed: 42], 300_000)
  end},

  {"high-p sparse (1K×100)", fn ->
    {x, y} = EnduranceBench.sparse_linear(1000, 100)
    EnduranceBench.run_test("high-p-100", x, y,
      [num_trees: 200, num_gfr: 10, num_mcmc: 100, seed: 42], 30_000)
  end},

  {"extreme sparse (5K×500)", fn ->
    {x, y} = EnduranceBench.sparse_linear(5000, 500)
    EnduranceBench.run_test("extreme-p-500", x, y,
      [num_trees: 200, num_gfr: 10, num_mcmc: 50, seed: 42], 120_000)
  end},

  {"smooth sin (5K×5)", fn ->
    {x, y} = EnduranceBench.smooth_sin(5000)
    EnduranceBench.run_test("smooth-sin", x, y,
      [num_trees: 200, num_gfr: 15, num_mcmc: 200, seed: 42], 60_000)
  end},

  {"california (20K×8)", fn ->
    {x, y} = EnduranceBench.california_housing()
    EnduranceBench.run_test("california", x, y,
      [num_trees: 200, num_gfr: 10, num_mcmc: 100, seed: 42], 120_000)
  end},

  {"multi-chain (5K×10, 4 chains)", fn ->
    {x, y} = EnduranceBench.friedman1(5000, 5)
    IO.write("  #{String.pad_trailing("multi-chain-4", 25)}")

    {us, results} = :timer.tc(fn ->
      1..4
      |> Task.async_stream(fn seed ->
        StochTree.BART.fit(x, y,
          num_trees: 200, num_gfr: 10, num_mcmc: 100, seed: seed * 1000)
      end, max_concurrency: 4, timeout: 600_000)
      |> Enum.map(fn {:ok, result} -> result end)
    end)

    ms = div(us, 1000)
    per_chain = Enum.map(results, fn {_forest, sigmas} ->
      Enum.sum(sigmas) / length(sigmas)
    end)

    IO.puts("#{ms}ms  (4 chains, #{div(ms, 4)}ms avg/chain)  sigma=#{inspect(Enum.map(per_chain, &Float.round(&1, 4)))}")

    %{name: "multi-chain-4", ms: ms}
  end}
]

# --- Run ---

IO.puts("Test                      Time     RMSE      Sigma    TopFeat  Status")
IO.puts(String.duplicate("-", 70))

results =
  if quick_mode do
    [{name, fun} | _] = tests
    [fun.()]
  else
    Enum.map(tests, fn {name, fun} ->
      try do
        fun.()
      rescue
        e -> IO.puts("  #{name}: FAILED — #{Exception.message(e)}")
        %{name: name, ms: 0, rmse: 0, error: true}
      end
    end)
  end

IO.puts("")
IO.puts("=" |> String.duplicate(70))
total_ms = Enum.sum(Enum.map(results, fn r -> Map.get(r, :ms, 0) end))
IO.puts("  Total: #{div(total_ms, 1000)}s")
IO.puts("=" |> String.duplicate(70))
