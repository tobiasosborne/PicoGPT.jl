include(joinpath(@__DIR__, "../../scripts/common.jl"))
include(joinpath(@__DIR__, "../src/Loop.jl"))
using .Loop, Statistics

const SEEDS = (1, 2, 3)
const RECIPE = (; steps = 400, batch = 24, η = 0.02, ηmin = 0.002, warmup = 20,
                 λ = 0.01, clip = 1.0, β1 = 0.9, β2 = 0.95, ε = 1e-8)
const BIT_IDS = 1:2               # 1 encodes bit 0, 2 encodes bit 1

config(c) = (; c.V, c.D, c.H, c.L, c.T, c.F)
binary_words(T) = [1 + ((word >> (t - 1)) & 1) for t in 1:T, word in 0:(1 << T) - 1]
bit_xor(a, b) = 1 + xor(a - 1, b - 1)

"""One step of rule 60, with an implicit zero to the left of the sequence."""
function rule60_step(x)
    next = copy(x)
    next[2:end, :] .= bit_xor.(x[2:end, :], x[1:end-1, :])
    next
end

"""Lucas' theorem: C(k,j) is odd iff every set bit of j is also set in k."""
function lucas(x, k)
    y = ones(Int, size(x))
    T = size(x, 1)
    for j in 0:min(k, T - 1)
        (j & k) == j || continue
        @views y[1+j:end, :] .= bit_xor.(y[1+j:end, :], x[1:end-j, :])
    end
    y
end

"""Fit fresh random batches. Keep every pre-update loss and one post-update loss."""
function fit(make_batch, c, seed, opts; recipe = RECIPE, supervision = :last, objective_op = looploss, θ_initial = nothing)
    BLAS.set_num_threads(1)
    (; steps, batch, η, ηmin, warmup, λ, clip, β1, β2, ε) = recipe
    rng = Xoshiro(1000 + seed)      # separate the data stream from the parameter draw
    θ = isnothing(θ_initial) ? init(c, Float64; rng = Xoshiro(seed)) : copy(θ_initial)
    opt = AdamW(θ; β1, β2, ε, λ = decay(c, λ))
    objective(θ, x, y) = objective_op(θ, x, y, c; supervision, opts...)
    # Compile forward and pullback before timing and leave the training RNG untouched.
    _, pull = objective(θ, make_batch(copy(rng), batch)...)
    pull()

    losses = zeros(steps + 1)
    x, y = make_batch(rng, batch)
    elapsed = @elapsed begin
        for t in 1:steps
            losses[t], back = objective(θ, x, y)
            step_size = warmup_cosine(t; η, ηmin, warmup, steps)
            update!(opt, θ, clip!(back(), clip), step_size)
            t < steps && ((x, y) = make_batch(rng, batch))
        end
        losses[end] = first(objective(θ, x, y))
    end
    (; θ, losses, elapsed)
end

"""Independent initial/final loss on one fixed random evaluation set."""
function endpoint_losses(make_batch, c, seed, opts, fitted; supervision = :last, objective_op = looploss, θ_initial = nothing)
    x, y = make_batch(Xoshiro(6060), 128)
    measure(θ) = first(objective_op(θ, x, y, c; supervision, opts...))
    θ = isnothing(θ_initial) ? init(c; rng = Xoshiro(seed)) : θ_initial
    (; initial = measure(θ), final = measure(fitted.θ))
end

position_accuracy(a, b) = vec(mean(a .== b; dims = 2))
accuracy(a, b) = mean(a .== b)

function pass_predictions(θ, x, c, opts)
    Zs, _ = loopgpt(θ, x, c; readouts = :all, opts...)
    [predictions(Z, size(x, 1)) for Z in Zs]
end

function example_trace(θ, x, c, opts)
    trace = NamedTuple[]
    loopgpt(θ, x, c; trace, opts...)
    [(; frame.pass, X = tensor(frame.X),
        (frame.pass == 0 ? (;) : (; A = tensor(frame.A), Z = tensor(frame.Z),
                                   predictions = tensor(frame.predictions)))...,
        (hasproperty(frame, :boundary_attention) ?
             (; boundary_attention = tensor(frame.boundary_attention)) : (;))...)
     for frame in trace]
end

"""A one-block F sweep can match the untied budget to within half of 2D+1."""
function control_configs(c, K)
    untied = Config(; c.V, c.D, c.H, c.T, c.F, L = K)
    wider_F = c.F + round(Int, (nparams(untied) - nparams(c)) / (2c.D + 1))
    wide = Config(; c.V, c.D, c.H, c.T, L = 1, F = wider_F)
    (; untied, wide)
end

function save_experiment(name, result)
    path = joinpath(@__DIR__, "../artifacts", name * ".json")
    writejson(path, result)
    println("Wrote ", normpath(path))
end
