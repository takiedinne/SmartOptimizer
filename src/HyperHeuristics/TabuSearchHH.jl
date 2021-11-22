#***********************************************************************
"""
the choice function from [1]

[1]: Misir, M., Verbeeck, K., De Causmaecker, P., & Vanden Berghe, G. (2012). 
     Design and Analysis of an Evolutionary Selection Hyper-heuristic Framework. 1–20.
"""
function choiceFunction(LLHs::DataFrame, currentPhase::Integer, t_remain)
    w1, w2, w3, w4, w5 = 1, 1, 1, 1, 1
    choiceFuncScores = DataFrame(index=Array{Integer,1}(), score=[])
    LLH_index = 1
    b = (sum(LLHs.C_p_best) > 0) ? 1 : 0
    for LLH in eachrow(LLHs)
        # first we check if this LLH is already in Tabu List or it finishes the tabu tenure
        if LLH.tabuState && currentPhase >= LLH.nbrPhaseEnterTabu + LLH.tabuTenure
            # getting out from the tabu list
            LLH.tabuState = false
        elseif !LLH.tabuState
            #for those who not are in tabu we check if we get them in to the tabu list
            # and those which have recently gotten out from tabu list are not considred by this check
            # so we calculate the choice function value as in the paper [1]
             
            s = w1 * ((LLH.C_p_best+1)^2 * (t_remain/LLH.t_spent)) * b + 
                w2 * (LLH.f_p_imp / LLH.t_p_spent) - 
                w3 * (LLH.f_p_wrs / LLH.t_p_spent) +
                w4 * (LLH.f_imp / LLH.t_spent) -
                w5 * (LLH.f_wrs / LLH.t_spent)
            push!(choiceFuncScores, [LLH_index, s])
        end
        LLH_index += 1
    end
    #count the number of LLH that are still tabu
    lengthTabuList = sum(LLHs.tabuState)  #we don't count the LLHs recentlly gotten out
    sort!(choiceFuncScores, [:score])
    nbrOfLLHNonTabuList = nrow(choiceFuncScores)
    # avg is the number of LLH that will be excluded for the next phase
    # all the LLHs that are tabu we give them QI=1 and we sort the non tabu LLH and we give tabuLearningMechanism
    # QI=r corespand to their rank (the best one get QI = n , n-1, ..., 1)
    # if avg is smaller than number of LLHs that have already been tabu no LLHs will be excluded 
    # in this process the LLHs that have recently gotten out are not considred
    avg = ((nbrOfLLHNonTabuList + 1) / 2 * nbrOfLLHNonTabuList + lengthTabuList) / (nbrOfLLHNonTabuList + lengthTabuList)
    # here we put the LLH in the tabu list
    for i in 1:(avg-lengthTabuList)
        
        LLHIndex= choiceFuncScores.index[Integer(i)]
        # we check if we increase the tabu duration in case of if this LLH has been tabu for the befor last phase
        if LLHs.nbrPhaseEnterTabu[LLHIndex] != 0 && LLHs.nbrPhaseEnterTabu[LLHIndex] + LLHs.tabuTenure[LLHIndex] + 1 == currentPhase
            LLHs.tabuTenure[LLHIndex] += 1
        end
        LLHs.tabuState[LLHIndex] = true
        LLHs.nbrPhaseEnterTabu[LLHIndex] = currentPhase
    end
end

function defaultSelectionFunc(LLHs::DataFrame, t_total, t_remain)
    prob = zeros(nrow(LLHs))
    for i in 1:nrow(LLHs)
        if !LLHs.tabuState[i]
            tf = t_remain / t_total
            p = ((LLHs.C_best[i] + 1)/ (t_total - t_remain ))^(1 + 3 * tf^3)
            prob[i] = p
        end
    end
    #println(prob)
    # we pick one LLH
    roulette(prob,1)[1]
end
mutable struct TabuSearchHH <: HyperHeuristic
    method_name::String
    tabuLearningMechanism::Function
    selectionMechanism::Function
    moveAcceptance::MoveAcceptanceMechanism
    episodeSize::Integer
    archiveSize::Integer

    totalTimeAlowed::Integer #en second i will fix it to 10 min
    initialTabuDuration::Integer
    maxTabuDuration::Integer
end
TabuSearchHH(;AS=10, TLM=choiceFunction, SM=defaultSelectionFunc, TA=600, MA= NaiveAcceptance(),
                InitTD=1, MAXTD = 10, es=1) = TabuSearchHH("Tabu Search Hyper Heuristic",
                                                        TLM, SM, MA, es, AS, TA, InitTD, MAXTD)
mutable struct TabuSearchHHState{T} <:HH_State
    x_best::Array{T,1} #reference
    x::Array{T,1} #current solution
    x_best_fit
    x_fit
    # golobal fields because  i didin't find a way to do the inheritance between the types
    LLHs::DataFrame # if we want to use Tabu search so not allways the same LLHs list 
    archive::DataFrame

    phaseLength::Integer
    actualPhase::Integer
    currNbrIterInthisPhase::Integer
    startTime
end
function initial_HHstate(method::TabuSearchHH, problem::Problem{T}) where {T<:Number}
   #initialise global fields
   methods= loadAllLLH()
   archive= DataFrame(x=[], fit=Array{Float64,1}())
   x= copy(problem.x_initial)
   fit=problem.objective(x)
   push!(archive,[x,fit])
   phaseLength = 10 # iteration it is fixed in the first version and we will see next
   nbrLLHs = length(methods)
   LLHs= DataFrame(LLH = methods, #Low level heuristic
                    C_best = zeros(nbrLLHs), # nbr of best new solution found by the considred LLH
                    f_imp = zeros(nbrLLHs), # cummulative amount of improvement during all the search process
                    f_wrs = zeros(nbrLLHs), # cumulative amount of worsness during all the search process
                    C_p_best = zeros(nbrLLHs), # nbr of best new solution found by the considred LLH during the current phase
                    f_p_imp = zeros(nbrLLHs), # cummulative amount of improvement during the actuel phase
                    f_p_wrs= zeros(nbrLLHs),
                    tabuState = falses(nbrLLHs), # cummulative amount of improvement during the actuel phase
                    nbrPhaseEnterTabu = zeros(nbrLLHs),
                    tabuTenure = ones(nbrLLHs).*method.initialTabuDuration,
                    t_spent = zeros(nbrLLHs),
                    t_p_spent = zeros(nbrLLHs))    
    TabuSearchHHState(x, copy(x), fit, fit, LLHs, archive, phaseLength, 1, 0, time()), 1
end 

# here we have two choice 
# 1 -> we can at each call of this function we performe a whole episodeSize
# 2 -> we can performe one iteration ( i think i will prefer this one)
    
function update_HHState!(method::TabuSearchHH, problem::Problem, HHState::TabuSearchHHState, iteration)
    t_remain= method.totalTimeAlowed - time() + HHState.startTime 
    # first step checking if we are in a new episode so we update some parameters
    if HHState.currNbrIterInthisPhase >= HHState.phaseLength
        HHState.currNbrIterInthisPhase = 0
        HHState.actualPhase += 1
        # here we calculate the new tabu liste
        method.tabuLearningMechanism(HHState.LLHs, HHState.actualPhase, t_remain)
        # we reassign the phase parameters for each LLH
        nbrLLHs = nrow(HHState.LLHs)
        HHState.LLHs.t_p_spent = zeros(nbrLLHs)
        HHState.LLHs.C_p_best = zeros(nbrLLHs)
        HHState.LLHs.f_p_imp = zeros(nbrLLHs)
        HHState.LLHs.f_p_wrs= zeros(nbrLLHs)
    end
    # we calculate the probabilities and we pick one LLH to apply 
    currentLLHIndex = method.selectionMechanism(HHState.LLHs,method.totalTimeAlowed, t_remain)
    currentLLH = HHState.LLHs.LLH[currentLLHIndex]

    newSolution, performance = apply_LLH!([currentLLH], problem, method.episodeSize, HHState)
    newSolution, performance = newSolution[1], performance[1]
    # cause we've applied only one LLH
    # we update the time consming by the considred LLH
    HHState.LLHs.t_p_spent[currentLLHIndex] += performance.CPUTime
    HHState.LLHs.t_spent[currentLLHIndex] += performance.CPUTime
    #here we update the amount of improvement and the worsness
    if performance.ΔFitness < 0
        HHState.LLHs.f_imp[currentLLHIndex] -= performance.ΔFitness
        HHState.LLHs.f_p_imp[currentLLHIndex] -= performance.ΔFitness
    else
        HHState.LLHs.f_wrs[currentLLHIndex] += performance.ΔFitness
        HHState.LLHs.f_p_wrs[currentLLHIndex] += performance.ΔFitness
    end
    # new solution is tuple of solution and fitness
    # move acceptance
    if isAccepted(method.moveAcceptance, newSolution, [HHState.x, HHState.x_fit])
        HHState.x, HHState.x_fit = newSolution
        #check if it is a new best solution
        if HHState.x_best_fit > newSolution[2]
            HHState.x_best, HHState.x_best_fit = newSolution
            HHState.LLHs.C_best[currentLLHIndex] += 1
            HHState.LLHs.C_p_best[currentLLHIndex] += 1
        end
    end
    #println(HHState.LLHs[!, [:C_best, :C_p_best]])
    HHState.currNbrIterInthisPhase += 1
    update_archive!(method, HHState, newSolution)
    HHState.x , HHState.x_fit, performance.numSimRun
end 
