# julia --project -t 12 loop/scripts/shakespeare_export.jl
include("shakespeare_common.jl")
using SHA

# Same passage as ATTENTION_PROMPT in scripts/export_book_data.jl.
const ATTENTION_PROMPT = "GREMIO:\nGood morrow, neighbour Baptista.\n\nBAPTISTA:\nGood morrow, neighbour Gremio.\n"
const SAMPLE_PROMPT = "\n"
const SAMPLE_LENGTH = 240
const SAMPLE_SEED = 2026
const SAMPLE_PASSES = (1, 2, 4, 8)
const MAX_PASS = 16
const BOOK1_LOSS = 1.5313

function read_tsv(path)
    header, rows... = readlines(path)
    keys = Tuple(Symbol.(split(header, '\t')))
    map(rows) do row
        values = map(zip(keys, split(row, '\t'))) do (key, value)
            key in (:step, :tokens, :passes) ? parse(Int, value) : parse(Float64, value)
        end
        NamedTuple{keys}(Tuple(values))
    end
end

function sample_text(θ, c, abc, opts, K)
    x = encode(SAMPLE_PROMPT, abc)
    rng = Xoshiro(SAMPLE_SEED)
    for _ in 1:SAMPLE_LENGTH
        context = view(x, max(1, length(x) - c.T + 1):length(x))
        Z, _ = windowgpt(θ, reshape(context, :, 1), c; K, opts...)
        push!(x, draw(rng, softmax(view(Z, :, size(Z, 2)))))
    end
    (; pass = K, text = decode(x, abc))
end

function state_settling(θ, c, abc, val, opts)
    x = reshape(encode(ATTENTION_PROMPT, abc), :, 1)
    start = findfirst(!=(encode("\n", abc)[1]), val)
    stop = start + length(x) - 1
    val[start:stop] == vec(x) || error("the attention passage does not match the held-out tail")
    steps, predictions = NamedTuple[], Vector{Int}[]
    visit_passes(θ, x, c; K = MAX_PASS, opts...) do k, Z, X, previous, _
        diff = Float64.(X) - Float64.(previous)
        centred = diff .- sum(diff; dims = 1) ./ size(diff, 1)
        push!(steps, (; from = k - 1, to = k, frobenius = norm(diff),
                      column_centred_frobenius = norm(centred),
                      column_l2 = norm.(eachcol(diff)),
                      column_centred_l2 = norm.(eachcol(centred))))
        push!(predictions, argmax.(eachcol(Z)))
    end
    by_position = map(1:length(x)) do t
        ids = [p[t] for p in predictions]
        last_change = findlast(k -> ids[k] != ids[k-1], 2:MAX_PASS)
        # findlast returns the index into 2:MAX_PASS, not its value.
        stable_from = isnothing(last_change) ? 1 : last_change + 1
        (position = t, input = abc[x[t]], target = abc[val[start + t]],
         stable_from_pass = stable_from, predictions = decode(ids, abc))
    end
    (; prompt = ATTENTION_PROMPT, validation_start = start, steps,
     positions = by_position, through_pass = MAX_PASS)
end

function cone_probe(θ, c, abc, val, starts, opts)
    window_positions = ((1, 33), (86, 128), (256, 231))
    alternatives = encode(" e", abc)
    checks = map(window_positions) do (window, t)
        start = starts[window]
        x = reshape(val[start:start+c.T-1], :, 1)
        function logits_at_query(input)
            logits = Vector{Float32}[]
            visit_passes(θ, input, c; K = MAX_PASS, opts...) do _, Z, _, _, _
                push!(logits, Z[:, t])
            end
            logits
        end
        baseline = logits_at_query(x)
        function perturb(position)
            changed = copy(x)
            before = changed[position]
            after = before == alternatives[1] ? alternatives[2] : alternatives[1]
            changed[position] = after
            logits = logits_at_query(changed)
            (; position, input = abc[before], replacement = abc[after],
             max_logit_deltas = [maximum(abs.(Float64.(a) - Float64.(b)))
                                 for (a, b) in zip(logits, baseline)])
        end
        probes = [perturb(position) for position in t-1:-1:t-MAX_PASS-2]
        evidence = map(1:MAX_PASS) do k
            outside, farther, edge, near = probes[k+1], probes[k+2], probes[k], probes[1]
            @assert iszero(outside.max_logit_deltas[k]) && iszero(farther.max_logit_deltas[k])
            (; pass = k, left_edge = t - k,
             outside_edge_delta = outside.max_logit_deltas[k], farther_delta = farther.max_logit_deltas[k],
             left_edge_delta = edge.max_logit_deltas[k], predecessor_delta = near.max_logit_deltas[k])
        end
        (; window, start, query = t, probes, evidence)
    end
    (; metric = "Maximum absolute logit difference over the vocabulary", checks)
end

function export_trial(model)
    recipe = MODELS[model]
    (; abc, train, val) = trial_data()
    path = run_dir(model)
    history = read_tsv(joinpath(path, "history.tsv"))
    updates = read_tsv(joinpath(path, "updates.tsv"))
    last(history).step == length(updates) == SETTINGS.steps || error("incomplete trial: $model")
    best = argmin(r -> r.val_loss, history)
    ck = load_checkpoint(joinpath(path, @sprintf("step_%06d.bin", best.step)))
    (; c, θ) = ck
    ck.abc == abc || error("checkpoint alphabet differs from the data")
    opts = model_options(recipe, c)
    val_eval = evalset(val, 2, c)
    val_loss = per_pass_loss(θ, val, val_eval, c, opts, MAX_PASS)
    isapprox(val_loss[recipe.eval_pass], best.val_loss; atol = 1e-6) ||
        error("pass evaluation does not match the checkpoint selection loss")
    samples = [sample_text(θ, c, abc, opts, k) for k in SAMPLE_PASSES]
    settling = state_settling(θ, c, abc, val, opts)
    diagnostics = model == :window ? (; cone = cone_probe(θ, c, abc, val, val_eval, opts)) : (;)
    profile_path = joinpath(path, "power-profile.txt")
    profile = isfile(profile_path) ? (; power_profile = readlines(profile_path)) : (;)
    pass_counts = [(pass = k, updates = count(r -> r.passes == k, updates)) for k in recipe.passes]
    block_applications = sum(r -> r.passes, updates)
    artifact_path = joinpath(TRIAL_ARTIFACTS, "shakespeare-$model.json")
    writejson(artifact_path, (
        model, config = (; c.V, c.D, c.H, c.L, c.T, c.F, parameters = nparams(c), precision = "Float32"),
        recipe = (; opts..., recipe.supervision, training_passes = collect(recipe.passes),
                  selection_pass = recipe.eval_pass),
        training = (; SETTINGS..., run_settings = chomp(read(joinpath(path, "settings.txt"), String)), profile...),
        split = (train_tokens = length(train), val_tokens = length(val), alphabet = String(abc),
                 sha256 = open(io -> bytes2hex(sha256(io)), artifact("data", "tinyshakespeare.txt"))),
        evaluation = (train_starts = evalset(train, 1, c), val_starts = val_eval,
                      book1_val_loss = BOOK1_LOSS, selected_step = best.step,
                      final_step = last(history).step, val_loss),
        compute = (; pass_counts, block_applications, mean_passes = block_applications / length(updates),
                   token_block_applications = block_applications * SETTINGS.B * c.T,
                   update_seconds = last(history).update_seconds, wall_seconds = last(history).wall_seconds),
        history, updates, samples = (prompt = SAMPLE_PROMPT, new_characters = SAMPLE_LENGTH,
                                     temperature = 1, seed = SAMPLE_SEED, samples), settling, diagnostics...,
    ); sig = 10)
    @printf("%s: step %d, pass %d, validation loss %.6f → %s\n",
            model, best.step, recipe.eval_pass, best.val_loss, artifact_path)
end

function main(models)
    BLAS.set_num_threads(1)
    foreach(export_trial, models)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(isempty(ARGS) ? (:one, :looped, :stock) : Symbol.(ARGS))
