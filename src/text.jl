# Chapter 1 · Text as a sequence of symbols
#
# A text is a finite sequence over a finite alphabet. We number the symbols of the
# alphabet 1…V, in sorted order, and from then on work only with the numbers.

"""
    alphabet(text) -> Vector{Char}

The distinct characters of `text`, sorted. Character `alphabet[a]` is symbol a ∈ 1:V.
"""
alphabet(text) = sort(unique(text))

"""
    encode(text, alphabet) -> Vector{Int}

The text as a sequence of symbol numbers.
"""
function encode(text, alphabet)
    number = Dict(ch => a for (a, ch) in enumerate(alphabet))
    [number[ch] for ch in text]
end

"""
    decode(x, alphabet) -> String

The inverse of `encode`.
"""
decode(x, alphabet) = String(alphabet[x])
