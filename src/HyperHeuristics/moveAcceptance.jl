abstract type MoveAcceptanceMechanism end


#see the paper hyper heuristic based reinforcement learning 2018.pdf
# be carful all this functions must have the same inputs
mutable struct OnlyImprovement <:MoveAcceptanceMechanism
    method_name::String
end
OnlyImprovement() = OnlyImprovement("Only improvement")
function MoveAcceptance(method::OnlyImprovement, newSolution, previouseSolution)
    return newSolution[2] < previousSolution[2]
end

mutable struct AllMoves <:MoveAcceptanceMechanism
    method_name::String
end
AllMoves() = AllMoves("All Moves")
function MoveAcceptance(method::AllMoves, newSolution, previouseSolution)
    return true
end


mutable struct NaiveAcceptance <:MoveAcceptanceMechanism
    method_name::String
    WorstAccProb::Real # accepting worrsing solutions probability
end
NaiveAcceptance(;WorstAccProb=0.5) = NaiveAcceptance("Naive Acceptance", WorstAccProb)
function MoveAcceptance(method::NaiveAcceptance, newSolution, previousSolution)
    accept= false
    if newSolution[2] < previousSolution[2]
        accept = true
    elseif rand() < method.WorstAccProb
        accept = true
    end 
    accept
end


function MetropolisAcceptance(newSolution, previousSolution)
    
end

function SimulatedAnnealingMA(newSolution, previousSolution) 

end

function LateAcceptance(newSolution, previousSolution)
end 
# there is other move acceptance you can find them in the implementation section in the same paper.
""" move acceptance from Masir Thesis 2012 who worked with Greet """


mutable struct ILTA <: MoveAcceptanceMechanism
    method_name::String
    R::Real # range of fitness solution to accept the worst solution 
    K::Integer # predifined nbr of worsing solutions to accept the worsing solution
    w_iteration::Integer # worsing solution counter
end
ILTA(;R=1.005, K=10) = ILTA("iteration limited threshold accepting", R, K ,0)
function MoveAcceptance(method::ILTA, newSolution, previousSolution)
    answer = false
    if newSolution[2] < previousSolution[2]
        answer = true
        method.w_iteration = 0
    elseif newSolution[2] == previousSolution[2]
        answer = true
    else
        method.w_iteration += 1
        if method.w_iteration > method.K && newSolution[2] < previousSolution[2] * method.R
            answer = true
            method.w_iteration = 0
        end
    end
    answer
end
