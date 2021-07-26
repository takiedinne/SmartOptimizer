include("src/SmartOptimizer.jl")
include("simulations/GG1K_simulation.jl")
include("simulations/BrainFuck/Brainfuck.jl")
include("simulations/BrainFuck/fitnessFunctions.jl")
using Main.Brainfuck
using Main.SmartOptimizer
using DataFrames
using CSV
p = Problem(fitnessStr,false,150, upper=Integer.(ones(150).*8), lower= Integer.(ones(150)))
x=p.x_initial
methods= DataFrame( abv=[], method=[], duration=[], x=[], fit=[], nbrSim=[] )

push!(methods, ["GA", GA(), 0, x, Inf, 0])
push!(methods, ["HJ", HookeAndJeeves(), 0, x, Inf, 0])
push!(methods, ["NM", NelderMead(), 0, x, Inf, 0])
push!(methods, ["PS", ParticleSwarm(), 0, x, Inf, 0])
push!(methods, ["SA", SimulatedAnnealing(), 0, x, Inf, 0])
push!(methods, ["SC", StochasticComparison(), 0, x, Inf, 0])
push!(methods, ["SR", StochasticRuler(), 0, x, Inf, 0])
push!(methods, ["TS", TabuSearch(), 0, x, Inf, 0])
push!(methods, ["GS", GeneratingSetSearcher(), 0, x, Inf, 0])
push!(methods, ["COMPASS", COMPASS_Searcher(), 0, x, Inf, 0])

t= time()
for i in 1:nrow(methods)
    method= methods[i, :method]
    res=0
    try
        res = optimize(method,p)
    catch e
        println("the ", method.method_name, " is interrupted...")
        msg = sprint(showerror, e)
        println(msg)
        read(stdin, Char)
        continue
    end
    if res != 0
        methods[i, :duration]=res[1].elapsed_time
        methods[i, :x]=res[1].minimizer
        methods[i, :fit]=res[1].minimum
        methods[i, :nbrSim]=res[2]
    end
end 


select!(methods, Not(:method))

total_time= time()-t

#CSV.write("C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\Experimental Results\\discrete low level heuristics comparison\\500Iter\\AllMethods500Iter.csv", methods)
#=
m= CSV.read("C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\Experimental Results\\discrete low level heuristics comparison\\AllMethods1000Iter.csv", DataFrame)
m.abv"""=#