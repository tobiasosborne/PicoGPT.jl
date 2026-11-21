include(joinpath(@__DIR__, "../../scripts/common.jl"))
include(joinpath(@__DIR__, "../src/window.jl"))
using .WindowLoop: windowgpt, windowloss, visit_passes

const TRIAL = joinpath(CHECKPOINTS, "loop-trial")
const TRIAL_ARTIFACTS = joinpath(@__DIR__, "../artifacts")
const SETTINGS = (D = 128, H = 4, L = 1, T = 256, F = 512, B = 32,
                  steps = 5000, warmup = 200, η = 1e-3, ηmin = 1e-4,
                  λ = 0.1, clip = 1.0, seed = 1234, eval_every = 250, eval_batches = 8)
const MODELS = (
    one = (injection = :none, start = :embedded, supervision = :last, passes = 1:1, eval_pass = 1),
    looped = (injection = :anchor_normalized, start = :boundary, supervision = :deep,
              passes = 3:5, eval_pass = 4),
    stock = (injection = :none, start = :embedded, supervision = :last, passes = 4:4, eval_pass = 4),
    fullstart = (injection = :anchor_normalized, start = :embedded, supervision = :deep,
                 passes = 3:5, eval_pass = 4),
    window = (injection = :anchor_normalized, start = :embedded, supervision = :deep,
              passes = 1:8, eval_pass = 8, width = 2),
)

run_dir(model) = joinpath(TRIAL, string(model))
model_options(recipe, c) = (; recipe.injection, recipe.start, position = :sinusoidal,
                            width = get(recipe, :width, c.T), lag_bias = 0)

function trial_data()
    text = read(artifact("data", "tinyshakespeare.txt"), String)
    abc = alphabet(text)
    train, val = shakespeare_split(text, abc)
    (; abc, train, val)
end

evalset(data, offset, c) =
    rand(Xoshiro(SETTINGS.seed + offset), 1:length(data) - c.T, SETTINGS.B * SETTINGS.eval_batches)

# Match Book 1's two-sequence grouping and fixed reduction order.
function sequence_groups(f, B)
    tasks = @sync [Threads.@spawn f(J) for J in Iterators.partition(1:B, 2)]
    fetch.(tasks)
end

function per_pass_loss(θ, data, starts, c, opts, K)
    x, y = windows(data, c.T, starts)
    parts = sequence_groups(size(x, 2)) do J
        losses = zeros(K)
        visit_passes(θ, view(x, :, J), c; K, opts...) do k, Z, _, _, _
            losses[k] = length(J) * first(crossentropy(Z, view(y, :, J)))
        end
        losses
    end
    sum(parts) / size(x, 2)
end

function selected_loss(θ, data, starts, c, opts, K)
    x, y = windows(data, c.T, starts)
    parts = sequence_groups(size(x, 2)) do J
        Z, _ = windowgpt(θ, x[:, J], c; K, opts...)
        length(J) * first(crossentropy(Z, y[:, J]))
    end
    sum(parts) / size(x, 2)
end
