#=
    Compass_kolda_neighborhood_structure from [1] altering only one diension
    [1] T. G. Kolda, R. M. Lewis, and V. Torczon, 
    “Optimization by direct search: New perspectives on some classical and modern methods,
    ” SIAM Rev., vol. 45, no. 3, pp. 385–482, 2003, doi: 10.1137/S003614450242889.
=#
function COMPASS_kolda_neighborhood_structure(x, upper, lower; step_size=1)
    neighbors = []
    for i in 1:length(x)
        x1 = copy(x)
        x2 = copy(x)
        x1[i] += 1 * step_size # move up
        x2[i] -= 1 * step_size# move down
        if x1[i] <= upper[i] 
            push!(neighbors,x1)
        end
        if x2[i] >= lower[]
            push!(neighbors,x2)
        end
    end
    neighbors
end

function COMPASS_promosing_area()
    # solve the linear inequality see how 
end


# default neighbors are those who are different from the original solution in one dimension
function defaultNeighborsGenerator(x, upbound, lowBound)  
    dim = length(x)
    neighbors=[[i == j ? x[i] + 1 : x[i] for i in 1:dim ] for j in 1:dim]
    append!(neighbors, [[i == j ? x[i] - 1 : x[i] for i in 1:dim ] for j in 1:dim])
    #neighbors
    return neighbors[[isFeasible(neighbors[i], upbound,lowBound) for i in eachindex(neighbors)]]
end


function Hamming_discrete_neighborhood_structure(x, upper, lower; step_size=1, neighbor_set_size = 5)
    #[1] E.-G. Talbi, METAHEURISTICS. 2009. page 90
    # this method change only one element in the individal basically it generate 
    # nbr of alphabet solutions
    neighbors = []
    variable_indices = shuffle!(collect(1:length(x)))
    for i in variable_indices
        for j in shuffle(lower[i]:upper[i])
            if j != x[i]
                x_trial = copy(x)
                x_trial[i] = j
                push!(neighbors, x_trial)
            end
        end
    end
    neighbors
end
