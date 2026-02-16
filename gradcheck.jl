# gradcheck.jl -- Finite-difference gradient verification utilities
#
# The physicist's unit check for neural networks: perturb each parameter by ±ε,
# measure the change in loss, compare with the analytical gradient.
#
# No dependencies beyond LinearAlgebra (already loaded by config.jl).

"""
    numerical_gradient(f, x; eps=1e-5)

Central-difference approximation of ∂f/∂x_i for every element of array `x`.
`f` must return a scalar.
"""
function numerical_gradient(f, x::AbstractArray{Float64}; eps::Float64=1e-5)
    grad = similar(x)
    for i in eachindex(x)
        old = x[i]
        x[i] = old + eps
        fp = f(x)
        x[i] = old - eps
        fm = f(x)
        x[i] = old
        grad[i] = (fp - fm) / (2 * eps)
    end
    grad
end

"""
    relative_error(a, b)

Max elementwise relative error: |a_i - b_i| / max(|a_i|, |b_i|, 1e-8).
"""
function relative_error(a::AbstractArray, b::AbstractArray)
    maximum(abs.(a .- b) ./ max.(abs.(a), abs.(b), 1e-8))
end

"""
    check_gradient(f_loss, analytical_grad, x; eps=1e-5, tol=1e-5)

Compare `analytical_grad` (an array the same shape as `x`) against the
finite-difference gradient of scalar function `f_loss(x)`.

Returns `(pass::Bool, rel_err::Float64)`.
"""
function check_gradient(f_loss, analytical_grad::AbstractArray, x::AbstractArray;
                        eps::Float64=1e-5, tol::Float64=1e-5)
    num_grad = numerical_gradient(f_loss, x; eps=eps)
    rel_err = relative_error(num_grad, analytical_grad)
    (rel_err < tol, rel_err)
end

"""
    check_gradient(f_loss, f_backward, x; eps=1e-5, tol=1e-5)

Convenience form: `f_backward(x)` is called to obtain the analytical gradient,
then compared against finite differences of `f_loss(x)`.

Returns `(pass::Bool, rel_err::Float64)`.
"""
function check_gradient(f_loss::Function, f_backward::Function, x::AbstractArray;
                        eps::Float64=1e-5, tol::Float64=1e-5)
    analytical_grad = f_backward(x)
    check_gradient(f_loss, analytical_grad, x; eps=eps, tol=tol)
end
