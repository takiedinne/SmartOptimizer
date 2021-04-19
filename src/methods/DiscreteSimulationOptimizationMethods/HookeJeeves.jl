include("../../../simulations/GG1K_simulation.jl")
#=
about the step size initially i will fixed to 
 0.25 from all the search space  and at each 
 failed we decrease this step size to the half 
=#
struct HookeAndJeeves
    initial_step_size::Real# percent from overall  the search space by default 25%
    step_reduction::Real # fraction of reduction by default 0.5
    ϵ_h::Real # here the small allowed step size by default 1
end
  
function HookeAndJeeves(;
initial_step_size = 0.25,
step_reduction = 0.5,
ϵ_h = 1.0)
return HookeAndJeeves(
    initial_step_size,
    step_reduction,
    ϵ_h
)
end

mutable struct HookeAndJeevesState{T}
    current_dim::Int
    step_size::Real
    f_k  #current scor
    x_k::Array{T,1}#current solution
    x_b::Array{T,1}
end

function initial_state(method::HookeAndJeeves,n::Integer,x_initial::Array{T,1},objfun::Function) where {T<:Number}
   
    return HookeAndJeevesState(
                1,
                method.initial_step_size,
                objfun(x_initial),
                copy(x_initial),
                copy(x_initial)
    )
end
  
n=3
f=sim_GG1K

function has_converged(method::HookeAndJeeves, state::HookeAndJeevesState) 
    # Convergence is based on step size
    return state.step_size < method.ϵ_h
end
function HookeJeevesAlgo(f, lower::Array{Int,1}, upper::Array{Int,1} ) 
    n= length(lower)
    initial_x=Array{Int,1}(undef,n)
    for i in 1:dim
        initial_x[i]=lower[i]+ round(rand()*(upper[i]-lower[i]))
    end

    method= HookeAndJeeves()

    state= initial_state(method,dim,initial_x,f)
    iterartion=0
    while iterartion<1000 && has_converged(method,state)
        iterartion+=1
        x_k, x_b = state.x_k, state.x_b
        # Evaluate a positive and a negative point in each cardinal direction
        # and update as soon as one is found
        while state.current_dim <= n
            
            # Arbitrarily choose the positive direction first
            improved=false
            for dir in [1,-1]
                #calculate step size
                length_step=round(state.step_size*(upper[state.current_dim]-lower[state.current_dim]))
                x_trial= copy( x_k)
                x_trial[state.current_dim] += dir * length_step

                #check if the new point in inbounds
                if x_trial[state.current_dim]>upper[state.current_dim]
                    x_trial[state.current_dim]=upper[state.current_dim]
                elseif x_trial[state.current_dim]<lower[state.current_dim]
                    x_trial[state.current_dim]=lower[state.current_dim]
                end

                f_trial = f(x_trial)
        
                # If the point is better, immediately go there
                if (f_trial <= state.f_k)
                    copy!(x_k, x_trial)
                    state.f_k = f_trial
                    state.current_dim += 1
                    improved=true
                    break
                end
            end
            improved && break
    
            state.current_dim += 1
        end
       
        # If the cardinal direction searches did not improve, reduce the
        # step size
        if (x_k == x_b)
            state.step_size *= method.step_reduction
        end
    
        # Attempt to move in an acceleration based direction
        x_trial = 2x_k - x_b
        f_trial = f(x_trial)
        copy!(x_b, x_k)
        state.current_dim = 1
    
        # If the point is an improvement use it
        if f_trial <= state.f_k
            copy!(x_k, x_trial)
            state.f_k = f_trial
        end
    end
    state
end
lower=[1,1,1]
upper=Int.(ones(3).*5)

s=HookeJeevesAlgo(sim_GG1K,lower,upper)

