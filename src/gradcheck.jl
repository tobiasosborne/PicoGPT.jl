# Chapter 7 · Pulling back, checked
#
# Every pullback in this package was derived by hand. Here is how we know they are right.

"""
    gradcheck(f, x, x̄; h = 1e-6) -> relative error

Compare a claimed gradient x̄ of the scalar function f at x with central differences,

    ∂f/∂xᵢ ≈ (f(x + h eᵢ) − f(x − h eᵢ)) / 2h        (error O(h²)),

and return |g − x̄| / max(|g|, |x̄|) for the estimate g. Use Float64.
"""
function gradcheck(f, x, x̄; h = 1e-6)
    x = copy(x)
    g = zeros(size(x))
    for i in eachindex(x)
        xᵢ = x[i]
        x[i] = xᵢ + h;  f₊ = f(x)
        x[i] = xᵢ - h;  f₋ = f(x)
        x[i] = xᵢ
        g[i] = (f₊ - f₋) / 2h
    end
    norm(g - x̄) / max(norm(g), norm(x̄))
end

"""
    checkpullback(op, args...; wrt) -> relative errors

The adjoint test, for any y, back = op(args...). For a random covector ȳ, the gradient of
the scalar function x ↦ ⟨ȳ, op(…, x, …)⟩ is Dop(x)ᵀ ȳ, which is exactly what back(ȳ)
claims to compute. Compare by `gradcheck`, for each argument position listed in `wrt`
(back returns the adjoints of those arguments, in that order).
"""
function checkpullback(op, args...; wrt)
    y, back = op(args...)
    ȳ = y isa Number ? randn() : randn(size(y))
    x̄s = back(ȳ)
    x̄s isa Tuple || (x̄s = (x̄s,))
    map(enumerate(wrt)) do (k, i)
        f(xᵢ) = dot(ȳ, first(op(ntuple(j -> j == i ? xᵢ : args[j], length(args))...)))
        gradcheck(f, args[i], x̄s[k])
    end
end
