using Hyperopt
using Plots
include("single_drone_opt_rev6.jl")
function hyper_opt_test()
   ho = @hyperopt for i in 10000,
      sampler in RandomSampler(), # This is default if none provided
      a in LinRange(1, 100, 100),
      b in LinRange(0.1, 10, 1000),
      c in LinRange(250, 1000, 100),
      d in LinRange(0.001, 0.04, 40)
      #cost = configuration(a, b, c, d)
      #print(i, "\t", a, "\t", b,"\t", c,"\t",d,  "   \t")
      #x = 100
      configuration(a, b, c, d)
      #f(a, b, c,d)
   end
   
   best_params, min_f = ho.minimizer, ho.minimum
   printmin(ho)
   #plot(ho)
end

# here it's my adaptation for the optimization methods.
# the objective function in this problem called Configuration(input1, input2, input3, input4)
# and return the Kenitic energy
# the first input is the nbr of fragements ∈ [1:100]
# the seconde inputs is distance of detonation ∈ [0.1, 10] (mitre)
# the third inputs is Projectile vilocity ∈ [250:1000] (mitre/s)
# the fourth input is the explosive charge ∈ [0.01, 0.4]
explosif_mass_range = collect(0.001:0.001:0.04) # 31 element
detonation_distance_range = collect(0.1:0.01:10) # 991 element

function single_drone(X)
   @assert length(X) == 4 "the problem need 4 parameters"
   number_of_fragments = Float64(X[1])
   detonation_distance = detonation_distance_range[X[2]]
   projectile_velocity = Float64(X[3])
   explosif_mass = explosif_mass_range[X[4]]
   configuration(number_of_fragments, detonation_distance, projectile_velocity, explosif_mass)
end