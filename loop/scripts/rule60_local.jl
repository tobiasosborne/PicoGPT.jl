# From the repository root: julia --project -O0 loop/scripts/rule60_local.jl
include(joinpath(@__DIR__, "rule60.jl"))
include(joinpath(@__DIR__, "structured_state.jl"))

function local_rule60_experiment(; cautious = false)
    c = Config(V = 2, D = 4, H = 1, L = 1, T = 6, F = 10)
    K = 3
    recipe = cautious ? CAUTIOUS_RECIPE : LOCAL_RECIPE
    perturbation = cautious ? CAUTIOUS_PERTURBATION : LOCAL_PERTURBATION
    runs = [
        run_rule60("local_spline", c, K; seed, recipe, transition = LOCAL_TRANSITION,
                   θ_initial = structured_init(c, :rule60; perturbation, seed),
                   horizon = 24, lengths = (c.T, 2c.T, 4c.T))
        for seed in SEEDS
    ]
    reference = if cautious
        θ = structured_init(c, :rule60)
        opts = (; injection = :none, LOCAL_TRANSITION...)
        evaluations = [evaluate_rule60(θ, c, opts, T, 24, K) for T in (c.T, 2c.T, 4c.T)]
        @assert all(all(p.lucas_accuracy == 1 for p in e.passes) for e in evaluations)
        impulse = ones(Int, 49, 1)
        impulse[1] = 2
        expected = [vec(lucas(impulse, k)) for k in 1:24]
        predicted = vec.(pass_predictions(θ, impulse, c, (; K = 24, opts...)))
        @assert expected == predicted
        (; θ, evaluations, impulse = (; input = vec(impulse), expected, predicted),
           one_pass_ceiling = (2 + (c.T - 2) / 2) / c.T)
    end
    name = cautious ? "rule60-local-cautious" : "rule60-local"
    save_experiment(name, (; task = "Local rule 60 with a ReLU spline initialization",
                              initial_perturbation = perturbation,
                              (cautious ? (; reference) : (;))..., runs))
end

abspath(PROGRAM_FILE) == (@__FILE__) && local_rule60_experiment(; cautious = "--cautious" in ARGS)
