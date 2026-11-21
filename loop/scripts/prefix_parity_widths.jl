# From the repository root: julia --project -O0 loop/scripts/prefix_parity_widths.jl
include(joinpath(@__DIR__, "prefix_parity.jl"))

function prefix_width_experiment()
    # Reuse the strongest stock seed to screen widths before investing in other seeds.
    seed = 2
    shapes = ((2, 4), (4, 4), (4, 12), (6, 8), (8, 4), (8, 8))
    runs = map(shapes) do (D, F)
        c = Config(V = 2, H = 1, L = 1, T = 5; D, F)
        run_prefix("tied_width", c, 5; seed, horizon = 6, lengths = (5, 10))
    end
    save_experiment("prefix-parity-widths", (; task = "Stock prefix XOR width screen",
                                               runs = collect(runs)))
end

abspath(PROGRAM_FILE) == (@__FILE__) && prefix_width_experiment()
