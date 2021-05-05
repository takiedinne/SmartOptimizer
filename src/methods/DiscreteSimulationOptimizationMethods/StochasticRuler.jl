
#= this algorithm is devlopped as described from
ALREFAEI, Mahmoud H. et ANDRADÓTTIR, Sigrún. 
Discrete stochastic optimization using 
variants of the stochastic ruler method. 
Naval Research Logistics (NRL), 2005, vol. 52, no 4, p. 344-360.=#
const M=3
struct StochasticRuler <:LowLevelHeuristic
    method_name::String
    M::Integer
end
StochasticRuler(;M=3)= StochasticRuler("Stochastic Ruler", M)
mutable struct StochasticRulerState{T} <: State
    x::Array{T,1}
    x_current::Array{T,1}
    f_x::Real
    f_current::Real
    upper_fit::Real
    lower_fit::Real
    A::Dict #fitnes mean for Xs
    C::Dict #nbr of viste
end
function initial_state(method::StochasticRuler, problem::Problem{T})where T
    upper_fit = 1000
    lower_fit = -1000
    f= problem.objective(problem.x_initial)
    A=Dict(problem.x_initial => f)
    C=Dict(problem.x_initial => 1)

    StochasticRulerState(problem.x_initial, copy(problem.x_initial), f, f, upper_fit, lower_fit, A, C)
end


function update_state!(method::StochasticRuler, problem::Problem{T}, iteration::Int, state::StochasticRulerState) where {T}
    A= state.A
    C = state.C
    dimToChange=rand(1:problem.dimension)
    changeValue= rand([-1,1])
    Z=copy(state.x_current)
    Z[dimToChange] += changeValue
    if problem.lower[dimToChange] <= Z[dimToChange] <= problem.upper[dimToChange]
        successInTest=true
        for i in 1:M
            fit_Z=problem.objective(Z)
            if haskey(A,Z)
                A[Z]=(A[Z]*C[Z] + fit_Z)/(C[Z]+1)
                C[Z]+=1
            else
                push!(A,Z=>fit_Z)
                push!(C,Z=>1)
            end
            stochastic_ruler=rand(state.lower_fit:state.upper_fit)
            if fit_Z > stochastic_ruler
                successInTest=false
                break
            end
        end
        if successInTest
            state.current=Z
        end
        #count the actual optimal 
        state.x = argmin(state.A)
        state.f_x = state.A[state.x]
       
    end
    
    state.x, state.f_x
end
function has_converged(method::StochasticRuler, x::Tuple{Array{T},Array{T}}, f::Tuple, options::Options, state::State) where {T<:Number}
    false
end