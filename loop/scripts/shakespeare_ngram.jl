# julia --project loop/scripts/shakespeare_ngram.jl
include("shakespeare_common.jl")
using SHA

# Token IDs fit in seven bits. A full 18-gram needs 126 bits.
const TOKEN_BITS = 7
const MAX_ORDER = 18
const DISCOUNTS = collect(1:99) ./ 100

suffix_mask(n) = (UInt128(1) << (TOKEN_BITS * n)) - 1

function ngram_counts(text, n)
    counts = Dict{UInt128,Int}()
    mask, key = suffix_mask(n), UInt128(0)
    for (i, token) in enumerate(text)
        key = ((key << TOKEN_BITS) | UInt128(token)) & mask
        i < n && continue
        counts[key] = get(counts, key, 0) + 1
    end
    counts
end

function continuation_counts(higher, n)
    counts = Dict{UInt128,Int}()
    mask = suffix_mask(n)
    for key in keys(higher)
        suffix = key & mask
        counts[suffix] = get(counts, suffix, 0) + 1
    end
    counts
end

# A zero key denotes an order that would cross the left edge of a window.
function prediction_keys(x, y, n)
    T = size(x, 1)
    mask = suffix_mask(n - 1)
    keys = zeros(UInt128, length(x))
    for b in axes(x, 2)
        context = UInt128(0)
        for t in axes(x, 1)
            context = ((context << TOKEN_BITS) | UInt128(x[t, b])) & mask
            t < n - 1 && continue
            keys[(b-1)*T + t] = (context << TOKEN_BITS) | UInt128(y[t, b])
        end
    end
    keys
end

function discount_terms(counts, queries)
    histories = Dict{UInt128,Tuple{Int,Int}}()
    for (key, count) in counts
        context = key >> TOKEN_BITS
        total, types = get(histories, context, (0, 0))
        histories[context] = (total + count, types + 1)
    end
    map(queries) do key
        total, types = get(histories, key >> TOKEN_BITS, (0, 0))
        (get(counts, key, 0), total, types)
    end
end

function interpolate(p, (count, total, types), discount)
    iszero(total) && return p
    (max(count - discount, 0.0) + discount * types * p) / total
end

function kn_ladder(text, x, y, discounts; max_order = MAX_ORDER, progress = false)
    @assert 3 ≤ max_order ≤ MAX_ORDER
    unigram = continuation_counts(ngram_counts(text, 2), 1)
    denominator = sum(values(unigram))
    base = [get(unigram, UInt128(token), 0) / denominator for token in vec(y)]
    lower = repeat(base, 1, length(discounts))
    loss = zeros(Float64, length(discounts), max_order - 2)
    coverage = NamedTuple[]

    for n in 3:max_order
        raw = ngram_counts(text, n)
        cont = continuation_counts(raw, n - 1)
        lower_terms = discount_terms(cont, prediction_keys(x, y, n - 1))
        top_keys = prediction_keys(x, y, n)
        top_terms = discount_terms(raw, top_keys)
        push!(coverage, (; full_context = count(!iszero, top_keys),
                         seen_context = count(term -> term[2] > 0, top_terms)))
        # The continuation probability through order n-1 is shared by top orders n and above.
        # Use the raw counts only when scoring the model whose top order is n.
        for (j, D) in enumerate(discounts)
            total_nll = 0.0
            @inbounds for i in eachindex(base)
                p = interpolate(lower[i, j], lower_terms[i], D)
                lower[i, j] = p
                total_nll -= log(interpolate(p, top_terms[i], D))
            end
            loss[j, n - 2] = total_nll / length(base)
        end
        progress && println("order $n: $(length(raw)) distinct training n-grams")
    end
    all(isfinite, loss) || error("nonfinite n-gram loss")
    (; loss, coverage)
end

function export_ngram()
    (; abc, train, val) = trial_data()
    @assert 1 ≤ length(abc) < 1 << TOKEN_BITS
    c = Config(; V = length(abc), SETTINGS.D, SETTINGS.H, SETTINGS.L, SETTINGS.T, SETTINGS.F)
    fit_end = floor(Int, 0.95 * length(train))
    fit, tune = view(train, 1:fit_end), view(train, fit_end+1:length(train))
    tune_offset = 3
    tune_starts = evalset(tune, tune_offset, c)
    x, y = windows(tune, c.T, tune_starts)
    println("Tuning discounts on the last $(length(tune)) characters of the training split")
    tuning_loss = kn_ladder(fit, x, y, DISCOUNTS; progress = true).loss
    chosen = argmin.(eachcol(tuning_loss))

    val_starts = evalset(val, 2, c)
    x, y = windows(val, c.T, val_starts)
    final_discounts = sort!(unique(DISCOUNTS[chosen]))
    println("Refitting on all $(length(train)) training characters")
    refit = kn_ladder(train, x, y, final_discounts; progress = true)
    ladder = map(1:MAX_ORDER-2) do k
        discount = DISCOUNTS[chosen[k]]
        (; pass = k, context = k + 1, order = k + 2, discount,
         tuning_loss = tuning_loss[chosen[k], k],
         val_loss = refit.loss[findfirst(==(discount), final_discounts), k],
         grid_loss = tuning_loss[:, k], refit.coverage[k]...)
    end

    path = joinpath(TRIAL_ARTIFACTS, "shakespeare-ngram.json")
    writejson(path, (
        estimator = "Interpolated Kneser–Ney with one absolute discount shared across orders within each model",
        counts = "Raw counts at the model order, distinct left-extension counts below it, continuation unigram base",
        boundaries = "No padding or sentence markers. Context starts at the window boundary and unavailable orders back off.",
        tuning = (; fit_range = (1, fit_end), slice_range = (fit_end + 1, length(train)),
                  window_seed = SETTINGS.seed + tune_offset, starts_in_slice = tune_starts,
                  discount_grid = DISCOUNTS, tie_break = "smallest discount"),
        split = (; train_tokens = length(train), val_tokens = length(val), alphabet = String(abc),
                 sha256 = open(io -> bytes2hex(sha256(io)), artifact("data", "tinyshakespeare.txt"))),
        evaluation = (; val_starts, window_length = c.T, predictions = length(y),
                       window_seed = SETTINGS.seed + 2, refit_tokens = length(train)),
        ladder,
    ); sig = 10)
    println(path)
end

abspath(PROGRAM_FILE) == (@__FILE__) && export_ngram()
