
function learningFunc()
end

function AdaptEpsilon()
    
end

mutable struct ϵGreedy <: HyperHeuristic
    ϵ::Real
    phaseSize::Integer
    learningFunction::Function
    adaptEpsilonFunc::Function
end
ϵGreedy(;ϵ=1, phaseSize=1 LF=learningFunc, aE=AdaptEpsilon)=ϵGreedy(ϵ, phaseSize, LF, aE)
mutable struct ϵGreedyState <: HH_State
    method_name::String
    LLHs::Array{LowLevelHeuristic,1} #if we want to use Tabu search so not allways the same LLHs list 
    x_best::Array{T,1} #reference
    x::Array{T,1} #current solution
    x_best_fit
    x_fit
    scores::Array{Float64, 1}
end

function initial_HHstate(method::ϵGreedy, problem::Problem{T}) where {T<:Number}
    #initialise LLHs
    x= copy(problem.x_initial)
    fit=problem.objective(x)
    LLHs= loadAllLLH()
    scores= ones(legnth(LLHs))

    ϵGreedyState( "ϵ-Greedy", LLHs, x, copy(x), fit, fit, scores)
end 

function update_HHState!(method::ϵGreedy, problem::Problem, HHState::ϵGreedyState, iteration)
    #selection mechanism
    random_number=rand()
    currentLLHIndex=0 
    if random_number < method.ϵ
        #random strategy
        currentLLHIndex=rand(1:length(HHState.LLHs))
        
    else
        #greedy strategy
        currentLLHIndex = argmax(HHState.scores)
    end
    currentLLH= HHState.LLHs[currentLLHIndex]
    #apply the selected LLH
    newSolution, performance = apply_LLH!(currentLLH, problem, method.phaseSize, HHState)
    #move acceptance
    if move_acceptance(newSolution, [HHState.x, HHState.x_fit])
        HHState.x = newSolution[1][1]
        HHState.x_fit = newSolution[1][2]
    end
    method.learningFunction(HHState, result, LHHs) # we will see later
        
end 
