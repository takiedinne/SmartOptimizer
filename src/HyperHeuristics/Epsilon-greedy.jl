function AdaptEpsilon()
end

mutable struct ϵGreedy <: HyperHeuristic
    method_name::String
    ϵ::Real
    phaseSize::Integer # must be in all the HH
    learningFunction::Function
    adaptEpsilonFunc::Function
    move_acceptance::Function
    archiveSize::Int
end
ϵGreedy(;ϵ=0.5, phaseSize=1, LF=fix_reward_punish, aE=AdaptEpsilon, MA=AllMoves, sizeArchive=10) =
 ϵGreedy("ϵ-Greedy",ϵ, phaseSize, LF, aE, MA, sizeArchive)

mutable struct ϵGreedyState{T} <: HH_State
    x_best::Array{T,1} #reference
    x::Array{T,1} #current solution
    x_best_fit
    x_fit
    scores::Array{Float64, 1}
    # golobal fields because  i didin't find a way to do the inheritance between the types
    LLHs::Array{LowLevelHeuristic,1} #if we want to use Tabu search so not allways the same LLHs list 
    archive::DataFrame
end

function initial_HHstate(method::ϵGreedy, problem::Problem{T}) where {T<:Number}
    #initialise global fields
    LLHs= loadAllLLH()
    archive= DataFrame(x=[], fit=Array{Float64,1}())
    x= copy(problem.x_initial)
    fit=problem.objective(x)
    push!(archive,[x,fit])
    scores= ones(length(LLHs)) # initially all the scores are equal
    ϵGreedyState( x, copy(x), fit, fit, scores, LLHs, archive)
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
    newSolution, performance = apply_LLH!([currentLLH], problem, method.phaseSize, HHState)
    #new solution is tuple of solution and fitness
    #move acceptance
    if method.move_acceptance(newSolution, [HHState.x, HHState.x_fit])
        HHState.x = newSolution[1][1]
        HHState.x_fit = newSolution[1][2]
    end
    method.learningFunction(HHState, performance, HHState.LLHs) # we will see later

    #update the archive
    update_archive!(method, HHState, newSolution[1])
    HHState.x , HHState.x_fit
end 
