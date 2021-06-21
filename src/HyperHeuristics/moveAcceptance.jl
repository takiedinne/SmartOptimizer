#see the paper hyper heuristic based reinforcement learning 2018.pdf
# be carful all this functions must have the same inputs
function OnlyImprovement(newSolution, previousSolution)
    return newSolution[2] < previousSolution[2]
end

function AllMoves(newSolution, previousSolution)
    return true
end

function NaiveAcceptance(newSolution, previousSolution)
    accept= false
    if newSolution[2] < previousSolution[2]
        accept = true
    elseif rand() < 0.5
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