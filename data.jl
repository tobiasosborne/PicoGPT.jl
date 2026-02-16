# data.jl -- Shakespeare dataset: download, tokenize, batch
#
# Character-level tokenization.  Sequential 90/10 train/val split.
# Non-overlapping windows chunked at startup.

"""
    download_shakespeare(path="input.txt") -> String

Download Shakespeare text from Karpathy's char-rnn repo if not already present.
Returns the text content.
"""
function download_shakespeare(path::String="input.txt")
    if !isfile(path)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        println("Downloading Shakespeare to $path ...")
        download(url, path)
    end
    read(path, String)
end

"""
    build_vocab(text) -> (char_to_idx, idx_to_char)

Character-level vocabulary from unique sorted characters.
1-indexed for Julia (char_to_idx maps Char -> Int, idx_to_char maps Int -> Char).
"""
function build_vocab(text::String)
    chars = sort(unique(text))
    char_to_idx = Dict(c => i for (i, c) in enumerate(chars))
    idx_to_char = Dict(i => c for (i, c) in enumerate(chars))
    (char_to_idx, idx_to_char)
end

"""
    encode(text, char_to_idx) -> Vector{Int}
"""
function encode(text::String, char_to_idx::Dict{Char,Int})
    [char_to_idx[c] for c in text]
end

"""
    decode(ids, idx_to_char) -> String
"""
function decode(ids::AbstractVector{Int}, idx_to_char::Dict{Int,Char})
    String([idx_to_char[i] for i in ids])
end

"""
    train_val_split(data; split_frac=0.9) -> (train_data, val_data)

Sequential split.  No shuffling -- avoids context-window leakage.
"""
function train_val_split(data::Vector{Int}; split_frac::Float64=0.9)
    n = length(data)
    split_idx = round(Int, n * split_frac)
    (data[1:split_idx], data[split_idx+1:end])
end

"""
    make_batches(data, seq_len, batch_size) -> Vector{Tuple{Matrix{Int}, Matrix{Int}}}

Non-overlapping windows of (seq_len + 1), grouped into batches.
Each batch is (input, target) where input[:,t] and target[t] = input[t+1].
input and target are both (T, B) = (seq_len, batch_size).
"""
function make_batches(data::Vector{Int}, seq_len::Int, batch_size::Int)
    window = seq_len + 1
    n_windows = div(length(data) - 1, window)  # -1 because we need seq_len+1 contiguous
    n_windows = div(length(data), window)
    # trim to exact multiple of batch_size
    n_usable = div(n_windows, batch_size) * batch_size

    batches = Tuple{Matrix{Int}, Matrix{Int}}[]
    for batch_start in 1:batch_size:n_usable
        input  = zeros(Int, seq_len, batch_size)
        target = zeros(Int, seq_len, batch_size)
        for b in 0:(batch_size-1)
            idx = (batch_start + b - 1) * window + 1
            chunk = data[idx:idx+window-1]
            input[:, b+1]  = chunk[1:seq_len]
            target[:, b+1] = chunk[2:seq_len+1]
        end
        push!(batches, (input, target))
    end
    batches
end
