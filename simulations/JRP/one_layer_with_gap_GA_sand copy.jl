using StaticArrays
using PolygonOps
using Polynomials:fit

using JLD2
using Plots

using Logging

using BenchmarkTools
import Base.show
using LinearAlgebra

using Evolutionary, Test, Random
include("../../src/SmartOptimizer.jl")
using Main.SmartOptimizer


function compute_bottom_factor(Nx::Int64, Ny::Int64; kwargs...)
    
    bottom_factor = fill("sand", (Ny, Nx))

    xv1 = [   0.,    0.,  1400.,  1400.,    0.] * Nx/1400.
    yv1 = [   0., 1400.,  1400.,     0.,    0.] * Ny/1400.
    

    polygon1 = SVector.(xv1,yv1)
    
    xa = 1:Nx
    ya = 1:Ny
    
    points = vec(SVector.(transpose(xa),ya))
    
    inside1 = [inpolygon(p, polygon1; in=true, on=true, out=false) for p in points]
    
    bottom_factor[inside1] .= "sand"   #0.7
    #bottom_factor[inside1] .= "ripples"   #0.4
    
    return bottom_factor
    
end

"""
    Grid

Grid structure using a mesh grid that covers the physical space. The area to be covered by the sensor 
is decomposed into small grid cells where the size of the cells is sufficiently small that the
coverage can be treated as uniform over the cell. 

### Fields:
- Lx: width of physical terrain [m]
- Ly: height of physical terrain [m]
- Lsquare: size of a grid cell [m]
- Nx: number of grid cells in x-direction
- Ny: number of grid cells in y-direction
"""
struct Grid
    
    Lx            ::Float64
    Ly            ::Float64
    Lsquare       ::Float64
    Nx            ::Int64
    Ny            ::Int64
    bottom_factor ::Array{String, 2}
    
    """
        Grid(Lx,Ly,lSquare; margin::Float64=0)

    Generate a grid from a physical rectangle of dimensions (Lx, Ly) making use a square 
    mesh with length lSquare.

    ### Notes
    no margin used (contrast with initial implementation)
    """
    function Grid(Lx::Float64, Ly::Float64, Lsquare::Float64; bottomfunction::Function=compute_bottom_factor, kwargs...)
        
        Nx = round(Int, Lx/Lsquare)
        Ny = round(Int, Ly/Lsquare)
        #bottom_factor =  ones(Float64, Ny, Nx)
        bottom_factor = bottomfunction(Nx, Ny; kwargs...)
        
        return new(Lx, Ly, Lsquare, Nx, Ny, bottom_factor)
        
    end
    
end


show(io::IO, g::Grid) = print(io, "A grid of size $(g.Nx)x$(g.Ny) cells")




"""
    Prob

Prob structure computes the probality of detection resulting from the current pass
or the cumulated probality of detections by taking into account previous (overlapping)
passes in addition to the current one.

### Fields:
- covered: cells that are covered by the current pass
- prob_local : probability of detection after current pass
- prob_global: cumulated probability of detection after previous (overlapping) passes
"""
mutable struct Prob

    Nx             ::Int64
    Ny             ::Int64


    covered        ::Array{Bool, 2}
    covered_global ::Array{Bool, 2}
    
    prob           ::Array{Float64, 2}
    prob_global    ::Array{Float64, 2}

    """
        Prob(g)

    Generates arrays storing covered cells and probabilities of detection (for current pass or previous ones).

    ### Notes
    no margin used (contrast with initial implementation)
    """
    function Prob(g::Grid)
        
        Nx = g.Nx
        Ny = g.Ny
    
        # set field equal to zeros rows for vertical, colums for horizontal
        covered        = zeros(Bool   , Ny, Nx)
        covered_global = zeros(Bool   , Ny, Nx)
        
        prob           = zeros(Float64, Ny, Nx)
        prob_global    = zeros(Float64, Ny, Nx)

        return new(Nx, Ny,covered, covered_global, prob, prob_global)
        
    end
    
end
    

show(io::IO, p::Prob) = print(io, "A proba of size $(p.Nx) x $(p.Ny) cells")


"""
    Sonar

The AUV collects sonar data from rmin to rmax away from the AUV in both 
the port and starboard directions. We assume sonar swath coverage is always 
achieved from rmin to rmax.The coverage performance is quantified by the 
lateral range curve. 

The lateral range curve defines the probability that a target at a specified 
lateral range from a sonar’s track will be detected along that track and 
subsequently correctly classified as being suggestive of a mine.

### Fields:
- rmin: minimal detection range [grid cells]
- rmax: maximal detection range [grid cells]
- lrc: lateral range curve profile
"""
struct Sonar
    
    rmin        ::Float64
    rmax        ::Float64

    """
        Sonar(rmin::Float64, rmax::Float64, lrc::Array{Float64,1})

    From the actual sonar characteristics [m] get the sonar range in [grid cells@]
    """
    function Sonar(rmin::Float64, rmax::Float64, g::Grid) 
        return new(rmin/g.Lsquare, rmax/g.Lsquare)
    end
    
end

show(io::IO, s::Sonar) = print(io, "A sonar with range $(s.rmin)-$(s.rmax)+1 cells")







"""
    gettrackparams(p1::Tuple(Float64, Float64), p2::Tuple(Float64, Float64))

from two points p1(x₁, y₁), p2(x₂, y₂) get the parameters from the equation of 
the straight line ax + by + c = 0 where a = -(y₂-y₁), b=(x₂-x₁) and 
c = -y₁(x₂ - x₁) + (y₂-y₁)x₁. Values are returned as a, b, c

See also  [`getintersections`](@ref)
"""


function gettrackparams(p1::Tuple{Float64, Float64}, p2::Tuple{Float64, Float64})
    return p1[2] - p2[2], p2[1] - p1[1], -p1[2]*(p2[1] - p1[1]) + p1[1]*(p2[2] - p1[2])
end




"""
    getintersections(a::Float64, b::Float64, c::Float64)

Given a grid limited by (1,1), (Nx, Ny) and a line defined by ax + by + c = 0,
determine the intersections with the grid limits.

Only the first two points within the grid are returned, sorted by x coordinate.

See also [`gettrackparams`](@ref), ['azimut'](@refs)
"""


function getintersections(a::Float64, b::Float64, c::Float64, g::Grid) 
    
    return unique(sort!(
            filter!(x-> ingrid(x, g), # limit to points in grid
                    [(1., -(a + c)/b);            # left limit
                     (Float64(g.Nx), -(a*g.Nx + c)/b);   # right limit
                     (-(b*g.Ny + c)/a, Float64(g.Ny));   # upper limit
                     (-(b + c)/a, 1.)]),      # lower limit 
                    by=x->x[1]))
end

function ingrid(x::Tuple{Float64, Float64}, g::Grid)
    return (1 <= x[1] <= g.Nx) && (1 <= x[2] <= g.Ny)
end




"""
    d(x::Float64,y::Float64,a::Float64, b::Float64, c::Float64)

Calculate the distance from a point (x,y) to a line defined by ax + by + c = 0
"""

d(x::Float64,y::Float64,a::Float64, b::Float64, c::Float64)::Float64 = abs(a*x + b*y + c) / sqrt(a^2 + b^2)




"""
f(x::Float64, y::Float64, s::Tuple{Float64, Float64}, t::Tuple{Float64, Float64})

For a point (x,y) in the domain, a track direction dir and a point p determine the 
the sign of the inner product product.
"""


function f(x::Float64, y::Float64, dir::Tuple{Float64, Float64}, p::Tuple{Float64,Float64})
    return dir[1] * (x - p[1]) + dir[2] * (y - p[2]) >= 0
end




"""
    singletrack!(p1::Tuple{Float64, Float64}, p2::Tuple{Float64, Float64}, g::Grid, s::Sonar, pdc::Prob)

Determine the detection probability for a single track with sonar s on grid g and proba pdc.
"""


function singletrack!(p1::Tuple{Float64, Float64}, p2::Tuple{Float64, Float64}, g::Grid, s::Sonar, pdc::Prob)
    
    @debug "single track  $(p1) -> $p2"
    # determine line properties
    dir = (p2[1]- p1[1], p2[2]-p1[2])
    a,b,c = gettrackparams(p1, p2)
    # intersection (currently not used)
    #inters = getintersections(a, b, c, g)
    # determine track
    
    #Threads.@threads for x in 1:g.Nx
    for x in 1:g.Nx
            for y in 1:g.Ny
            if s.rmin <= d(Float64(x),Float64(y), a, b, c) <= s.rmax
                if f(Float64(x),Float64(y),dir,p1) && !f(Float64(x),Float64(y),dir,p2)
                    
                    @inbounds pdc.covered[y,x] = true

                    if     g.bottom_factor[y,x] == "sand"
                        @inbounds pdc.prob[y,x] = lrc_sand(d(Float64(x),Float64(y), a, b, c),s)
                    elseif g.bottom_factor[y,x] == "ripples"
                        @inbounds pdc.prob[y,x] = lrc_ripples(d(Float64(x),Float64(y), a, b, c),s)
                    elseif g.bottom_factor[y,x] == "mud"
                        @inbounds pdc.prob[y,x] = lrc_mud(d(Float64(x),Float64(y), a, b, c),s)
                    else
                        @inbounds pdc.prob[y,x] = lrc_default(d(Float64(x),Float64(y), a, b, c),s)
                    end
                    
                else
                    @inbounds pdc.covered[y,x] = false
                    @inbounds pdc.prob[y,x] = 0.
                end
            else
                @inbounds pdc.covered[y,x] = false
                @inbounds pdc.prob[y,x] = 0.
            end
        end
    end
    
    pdc.covered_global .= max.(pdc.covered_global, pdc.covered)    
    
    # update global prob value
    pdc.prob_global .= pdc.prob_global .+(1.0 .-pdc.prob_global).*pdc.prob
end



function singletrack_horizontal!(d::Float64, g::Grid, s::Sonar, pdc::Prob)
    
    @debug "single track"

    pdc.covered        = zeros(Bool   , g.Ny, g.Nx)
    pdc.prob           = zeros(Float64, g.Ny, g.Nx)
    
    #Threads.@threads for x in 1:g.Nx
    for x in 1:g.Nx

        for y in max(1,round(Int, d-s.rmax+1)):min(round(Int, d-s.rmin),g.Ny)
            
            @inbounds pdc.covered[y,x] = true

            if     g.bottom_factor[y,x] == "sand"
                @inbounds pdc.prob[y,x] = lrc_sand(abs(Float64(y)-d),s)
            elseif g.bottom_factor[y,x] == "ripples"
                @inbounds pdc.prob[y,x] = lrc_ripples(abs(Float64(y)-d),s)
            elseif g.bottom_factor[y,x] == "mud"
                @inbounds pdc.prob[y,x] = lrc_mud(abs(Float64(y)-d),s)
            else
                @inbounds pdc.prob[y,x] = lrc_default(abs(Float64(y)-d),s)
            end                    

        end
        
        for y in max(1,round(Int, d+s.rmin+1)):min(round(Int, d+s.rmax),g.Ny)
            
            @inbounds pdc.covered[y,x] = true

            if     g.bottom_factor[y,x] == "sand"
                @inbounds pdc.prob[y,x] = lrc_sand(abs(Float64(y)-d),s)
            elseif g.bottom_factor[y,x] == "ripples"
                @inbounds pdc.prob[y,x] = lrc_ripples(abs(Float64(y)-d),s)
            elseif g.bottom_factor[y,x] == "mud"
                @inbounds pdc.prob[y,x] = lrc_mud(abs(Float64(y)-d),s)
            else
                @inbounds pdc.prob[y,x] = lrc_default(abs(Float64(y)-d),s)
            end                    

        end
        
        
        
        
    end
    
    pdc.covered_global .= max.(pdc.covered_global, pdc.covered)    
    
    # update global prob value
    pdc.prob_global .= pdc.prob_global .+(1.0 .-pdc.prob_global).*pdc.prob
end



"""
    multitrack!(waypoints::Array{Tuple{Float64, Float64},1}, g::Grid, s::Sonar, pdc::Prob)

From a list of waypoints, generate the sonar coverage

See also [`singletrack!`](@ref)
"""


function multitrack!(waypoints::Array{Tuple{Float64, Float64},2}, g::Grid, s::Sonar, pdc::Prob)
        
    @debug "multiple track "
    for i in 1:size(waypoints,1)
            # get individual contribution
            singletrack!(waypoints[i,1], waypoints[i,2], g::Grid, s::Sonar, pdc::Prob)
    end

    #minimize -fun = maximize fun
    return -sum(pdc.prob_global)/(pdc.Nx*pdc.Ny)    

end



function multitrack_horizontal!(d::Array{Float64,1}, g::Grid, s::Sonar, pdc::Prob)
        
    @debug "multiple track "
    for i in 1:length(d)
            # get individual contribution
            singletrack_horizontal!(sum(d[1:i]), g::Grid, s::Sonar, pdc::Prob)
    end

    #minimize -fun = maximize fun
    return -sum(pdc.prob_global)/(pdc.Nx*pdc.Ny)    

end




"""
    lawnmower_pattern(d::Float64)

Generate waypoints in a lawnmower pattern (vertical scan)
"""


function horizontal_pattern_vector(;d::Array{Float64,1}, g::Grid)

    waypoints_scan   = Array{Tuple{Float64, Float64},2}(undef,0,2)
    waypoints_travel = Array{Tuple{Float64, Float64},2}(undef,0,2)
    spacing          = Array{Float64}(undef,0)
    
    # complete list
    
    #d = filter(i -> i != 0., d)
    #######d = [d[1]; filter(i -> i != 0., d[2:end])]
    
    for i = 1:length(d)
        
        
        #if d[i] > 0. && sum(d[1:i]) <= Float64(g.Ny)
        if sum(d[1:i]) <= Float64(g.Ny)
            
                    waypoints_scan = vcat(waypoints_scan, [(0. ,  sum(d[1:i]))  (Float64(g.Nx),  sum(d[1:i]))])  
            
            if (mod(i,2) == 1)
                
                if (i == 1 )
                    
                    waypoints_travel = vcat(waypoints_travel, [(0. ,  sum(d[1:i  ]))   (Float64(g.Nx),  sum(d[1:i]))])
                
                else
                    
                    if d[i] != 0.
                        waypoints_travel = vcat(waypoints_travel, [(0.,   sum(d[1:i-1]))   (0.           ,  sum(d[1:i]))])              
                    end
                        
                    #waypoints_travel = vcat(waypoints_travel, [(0.,   sum(d[1:i-1]))   (0.           ,  sum(d[1:i]))])              
                    waypoints_travel = vcat(waypoints_travel, [(0. ,  sum(d[1:i  ]))   (Float64(g.Nx),  sum(d[1:i]))])               
                
                end
            
            else

                    if d[i] != 0.
                        waypoints_travel = vcat(waypoints_travel, [(Float64(g.Nx),  sum(d[1:i-1]))     (Float64(g.Nx),  sum(d[1:i]))])              
                    end
                
                    #waypoints_travel = vcat(waypoints_travel, [(Float64(g.Nx),  sum(d[1:i-1]))     (Float64(g.Nx),  sum(d[1:i]))])
                    waypoints_travel = vcat(waypoints_travel, [(Float64(g.Nx),  sum(d[1:i  ]))     (0.           ,  sum(d[1:i]))])               
                                   
            end
        
            spacing = vcat(spacing, d[i])

        end
    end
    
    return (waypoints_scan, waypoints_travel, spacing)

end



function getineqH(g)
    if g <=0
        H=0
    else
        H=1
    end
end



function fobj_horizontal_vector(x::Array{Float64,1}, g::Grid, s::Sonar, max_length::Float64 = 10.0^4)

        
    (waypoints_scan, waypoints_travel, spacing) = horizontal_pattern_vector(d=x, g=g)
    
    pdc = Prob(g)

    # determine the track and coverage
    y = multitrack_horizontal!(x, g, s, pdc) 
    
    #println("y =  $y")

    # Inequality constraints

    # Penalty constant
    lambda  = 10^15

    # Maximum travelling distance allowed
    #bound   = 3*10^4
         
    #ineq   = sum([norm(collect(row[1]) - collect(row[2])) for row in eachrow(waypoints_travel)]) 
    ineq1   = sum([norm(collect(row[1]) - collect(row[2])) for row in eachrow(waypoints_travel)]) - max_length
    ineq2   = sum(x) - g.Ly/g.Lsquare
    #ineq3   = g.Ly/g.Lsquare - s.rmax - sum(x) 


   
    return y + lambda*ineq1^2*getineqH(ineq1) + lambda*ineq2^2*getineqH(ineq2) 


end



function lrc_sand(range::Float64, s::Sonar)
    
    if s.rmin <= range <= s.rmax
        pd = 0.70
    else 
        pd = 0.0
    end
    
end

function lrc_ripples(range::Float64, s::Sonar)
    
    if s.rmin <= range <= s.rmax
        pd = 0.40
    else 
        pd = 0.0
    end
    
end

function lrc_mud(range::Float64, s::Sonar)
    
    if s.rmin <= range <= s.rmax
        pd = 0.25
    else 
        pd = 0.0
    end
    
end

function lrc_default(range::Float64, s::Sonar)
    
    if s.rmin <= range <= s.rmax
        pd = 1.0
    else 
        pd = 0.0
    end
    
end


# define grid settings
Lx = 1400.
Ly = 1400.

Lsquare = 10.

rmin = 40.
rmax = 140.

#make grid (new one at each time for illustrations)
g = Grid(Lx, Ly, Lsquare)

# make sonar (depends on grid)
s = Sonar(rmin, rmax, g)


lb = 0. #s.rmin*2  #0.
ub = s.rmax*2  #Ly

nvars = 5

max_length = (nvars+1)*Ly/Lsquare

#=
lower = fill(lb ,1, nvars-1) #this is because we append 0. to lower: lower = [0. lower]
upper = fill(ub ,1, nvars-1) #this is because we append Ly to upper: upper = [Ly upper]   
    
lower = [0. lower]
upper = [g.Ny upper];
=#


lower = lb .* ones(nvars) #this is because we append 0. to lower: lower = [0. lower]
upper = ub .* ones(nvars) #this is because we append Ly to upper: upper = [Ly upper]   


fobj_horizontal = x -> fobj_horizontal_vector(Float64.(x), g, s, max_length) 



p = Problem(fobj_horizontal, false, 5, upper = upper, lower = lower)
m = ParticleSwarm()

optimize(m, p)