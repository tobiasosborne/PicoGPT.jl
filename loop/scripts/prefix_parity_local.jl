# From the repository root: julia --project -O0 loop/scripts/prefix_parity_local.jl
include(joinpath(@__DIR__, "prefix_parity.jl"))
include(joinpath(@__DIR__, "structured_state.jl"))

function local_prefix_experiment(; cautious = false)
    c = Config(V = 2, D = 6, H = 1, L = 1, T = 5, F = 22)
    K = c.T - 1
    recipe = cautious ? CAUTIOUS_RECIPE : LOCAL_RECIPE
    perturbation = cautious ? CAUTIOUS_PERTURBATION : LOCAL_PERTURBATION
    runs = [
        run_prefix("local_spline", c, K; seed, recipe, transition = LOCAL_TRANSITION,
                   θ_initial = structured_init(c, :prefix; perturbation, seed),
                   horizon = 20)
        for seed in SEEDS
    ]
    reference = if cautious
        θ = structured_init(c, :prefix)
        opts = (; injection = :after, LOCAL_TRANSITION...)
        evaluations = [evaluate_prefix(θ, c, opts, T, 20) for T in (c.T, 2c.T, 4c.T)]
        @assert all(all(p.window_accuracy == 1 for p in e.passes) for e in evaluations)
        (; θ, evaluations, one_pass_ceiling = (2 + (c.T - 2) / 2) / c.T)
    end
    name = cautious ? "prefix-parity-local-cautious" : "prefix-parity-local"
    save_experiment(name, (; task = "Local prefix parity with a ReLU spline initialization",
                              initial_perturbation = perturbation,
                              (cautious ? (; reference) : (;))..., runs))
end

abspath(PROGRAM_FILE) == (@__FILE__) && local_prefix_experiment(; cautious = "--cautious" in ARGS)
