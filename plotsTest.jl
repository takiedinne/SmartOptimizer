

using Plots, StatsPlots, DataFrames

x= 1:10; y= rand(10,2)
x1= 1:10; y1= rand(10,2)
plot(x,y)
plot(x1,y1)
iteration = 1000
fit_historic=rand(1001)

anim = @animate for i ∈ 1:(iteration+1)
    plot(1:i, fit_historic[1:i])
end
gif(anim, "anim_fps15.gif", fps = 10)
a="aaa"
string(a, "kkk")