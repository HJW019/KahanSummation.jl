# Taken from https://github.com/JuliaMath/KahanSummation.jl/pull/7/files
#   and make the follwoing changes: (1) Comment out the first two lines
#   ("__precom.." and "module...") and comment out all the lines after Line
#   100. (2) Update the function name `mapreduce_single` to `mapreduce_first`.
# Use `include("KahanSummation_patch.jl")` to load the patch.

# This file contains code that was formerly a part of Julia.
# License is MIT: https://julialang.org/license

# __precompile__(true)
# module KahanSummation


if VERSION >= v"0.7.0-DEV.3000" # TODO: More specific bound
    if isdefined(Base, :sum_kbn) # Deprecated
        import Base: sum_kbn, cumsum_kbn
    else
        export sum_kbn, cumsum_kbn
    end
end
if isdefined(Base, Symbol("@default_eltype"))
    using Base: @default_eltype
else
    macro default_eltype(itr)
        quote
            Core.Inference.return_type(first, Tuple{$(esc(itr))})
        end
    end
end
if isdefined(Base, :promote_sys_size_add)
    using Base: promote_sys_size_add
else
    promote_sys_size_add(x::T) where {T} = Base.r_promote(+, zero(T)::T)
end

"""
    TwicePrecisionN{T}
Represents an extended precision number as `x.hi - x.nlo`.
We store the lower order component as the negation to avoid problems when `x.hi == -0.0`.
"""
struct TwicePrecisionN{T}
    hi::T
    nlo::T
end


@inline function plus_kbn(x::T, y::T) where {T}
    hi = x + y
    nlo = abs(x) > abs(y) ? (hi - x ) - y : (hi - y) - x
    TwicePrecisionN(hi, nlo)
end
@inline function plus_kbn(x::T, y::TwicePrecisionN{T}) where {T}
    hi = x + y.hi
    if abs(x) > abs(y.hi)
        nlo = ((hi - x) - y.hi) + y.nlo
    else
        nlo = ((hi - y.hi) - x) + y.nlo
    end
    TwicePrecisionN(hi, nlo)
end
@inline plus_kbn(x::TwicePrecisionN{T}, y::T) where {T} = plus_kbn(y, x)

@inline function plus_kbn(x::TwicePrecisionN{T}, y::TwicePrecisionN{T}) where {T}
    hi = x.hi + y.hi
    if abs(x.hi) > abs(y.hi)
        nlo = (((hi - x.hi) - y.hi) + y.nlo) + x.nlo
    else
        nlo = (((hi - y.hi) - x.hi) + x.nlo) + y.nlo
    end
    TwicePrecisionN(hi, nlo)
end

Base.convert(::Type{TwicePrecisionN{T}}, x::Number) where {T} =
    TwicePrecisionN{T}(convert(T, x), zero(T))
Base.convert(::Type{T}, x::TwicePrecisionN) where {T} =
    convert(T, x.hi - x.nlo)

@static if VERSION >= v"0.7.0-"
    Base.mapreduce_empty(f, ::typeof(plus_kbn), T) = TwicePrecisionN(zero(T),zero(T))
    Base.mapreduce_empty(::typeof(identity), ::typeof(plus_kbn), T) = TwicePrecisionN(zero(T),zero(T)) # disambiguate
    Base.mapreduce_first(f, ::typeof(plus_kbn), x) = TwicePrecisionN(x, zero(x))
else
    Base.r_promote_type(::typeof(plus_kbn), ::Type{T}) where {T} =
        TwicePrecisionN{T}
    Base.mr_empty(f, ::typeof(plus_kbn), T) = TwicePrecisionN(zero(T),zero(T))
end

singleprec(x::TwicePrecisionN{T}) where {T} = convert(T, x)


"""
    sum_kbn([f,] A)
Return the sum of all elements of `A`, using the Kahan-Babuska-Neumaier compensated
summation algorithm for additional accuracy.
"""
sum_kbn(f, X) = singleprec(mapreduce(f, plus_kbn, X))
sum_kbn(X) = sum_kbn(identity, X)




#=


"""
    cumsum_kbn(A, dim::Integer)
@@ -85,32 +157,4 @@ function cumsum_kbn(v::AbstractVector{T}) where T<:AbstractFloat
    return r
end

"""
    sum_kbn(A)
Return the sum of all elements of `A`, using the Kahan-Babuska-Neumaier compensated
summation algorithm for additional accuracy.
"""
function sum_kbn(A)
    T = @default_eltype(typeof(A))
    c = promote_sys_size_add(zero(T)::T)
    i = start(A)
    if done(A, i)
        return c
    end
    Ai, i = next(A, i)
    s = Ai - c
    while !(done(A, i))
        Ai, i = next(A, i)
        t = s + Ai
        if abs(s) >= abs(Ai)
            c -= ((s-t) + Ai)
        else
            c -= ((Ai-t) + s)
        end
        s = t
    end
    s - c
end

end # module
=#
