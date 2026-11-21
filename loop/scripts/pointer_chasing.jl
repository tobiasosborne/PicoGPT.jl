# From the repository root: julia --project loop/scripts/pointer_chasing.jl
include(joinpath(@__DIR__, "common.jl"))

permutations(xs) = isempty(xs) ? [Int[]] :
    [[x; rest] for x in xs for rest in permutations(filter(!=(x), xs))]

function pointers(π, order, start, K)
    n = length(π)
    x = [n * (i - 1) + π[i] for i in order]
    push!(x, n^2 + start)
    hops = Int[]
    for _ in 1:K
        start = π[start]
        push!(hops, n^2 + start)
    end
    x, hops
end

"""Score only the final query column in each sequence and scatter its adjoint back."""
function query_loss(θ, x, y, c; supervision = :last, opts...)
    supervision == :last || throw(ArgumentError("the pointer objective scores the final pass"))
    T = size(x, 1)
    Z, back_Z = loopgpt(θ, x, c; opts...)
    queries = T:T:size(Z, 2)
    ℓ, back_ℓ = crossentropy(Z[:, queries], y)
    function back(ℓ̄ = 1)
        Z̄ = zero(Z)
        Z̄[:, queries] .= back_ℓ(ℓ̄)
        back_Z(Z̄)
    end
    ℓ, back
end

function pointer_batch(worlds, n, K)
    function make_batch(rng, B)
        x = Matrix{Int}(undef, n + 1, B)
        y = Vector{Int}(undef, B)
        for b in 1:B
            column, hops = pointers(rand(rng, worlds), randperm(rng, n), rand(rng, 1:n), K)
            x[:, b] = column
            y[b] = last(hops)
        end
        x, y
    end
end

function evaluate_pointers(θ, c, opts, worlds, n, target_k, eval_K)
    orders = permutations(collect(1:n))
    examples = [pointers(π, order, start, max(eval_K, target_k))
                for π in worlds for order in orders for start in 1:n]
    correct = zeros(Int, eval_K)
    correct_target = zeros(Int, eval_K)
    correct_input = 0
    for batch in Iterators.partition(examples, 64)
        x = stack(first.(batch))
        y = stack(last.(batch))
        ps = pass_predictions(θ, x, c, (; opts..., K = eval_K))
        for k in 1:eval_K
            correct[k] += count(ps[k][end, :] .== y[k, :])
            correct_target[k] += count(ps[k][end, :] .== y[target_k, :])
        end
        correct_input += count(x[end, :] .== y[target_k, :])
    end
    (; examples = length(examples), input_only_accuracy = correct_input / length(examples),
       passes = [(; pass = k, hop_accuracy = correct[k] / length(examples),
                  trained_target_accuracy = correct_target[k] / length(examples)) for k in 1:eval_K])
end

function pointer_experiment()
    n = 4
    c = Config(V = n^2 + n, D = 12, H = 2, L = 1, T = n+1, F = 24)
    target_k = 2
    worlds = permutations(collect(1:n))
    make_batch = pointer_batch(worlds, n, target_k)
    recipe = (; RECIPE..., batch = 32, steps = 500)
    runs = map(Iterators.product((1, target_k), SEEDS)) do (K, seed)
        opts = (; K, tied = true, injection = :after)
        fitted = fit(make_batch, c, seed, opts; recipe, objective_op = query_loss)
        eval = evaluate_pointers(fitted.θ, c, opts, worlds, n, target_k, 4)
        endpoint = endpoint_losses(make_batch, c, seed, opts, fitted; objective_op = query_loss)
        name = K == 1 ? "one_pass" : "tied"
        @printf("%s seed %d P=%d loss %.4f → %.4f target acc %.4f %.2fs\n",
                name, seed, nparams(c), endpoint.initial, endpoint.final,
                eval.passes[K].trained_target_accuracy, fitted.elapsed)
        (; name, seed, config = config(c), nparams = nparams(c), options = opts,
           recipe, update_loss = fitted.losses, endpoint, train_seconds = fitted.elapsed,
           evaluation = eval, θ = fitted.θ)
    end
    save_experiment("pointer-chasing", (; task = "Two permutation hops", symbols = n,
                                          target_k, runs = vec(runs)))
end

abspath(PROGRAM_FILE) == (@__FILE__) && pointer_experiment()
