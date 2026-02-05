using Algames
using StaticArrays
using LinearAlgebra

T = Float64

# Define the dynamics of the system
p = 4 # Number of players
model = UnicycleGame(p=p) # game with 3 players with unicycle dynamics
n = model.n
m = model.m

# Define the horizon of the problem
N = 80 # N time steps
dt = 0.1 # each step lasts 0.1 second
probsize = ProblemSize(N,model) # Structure holding the relevant sizes of the problem

# Define the objective of each player
Q = [Diagonal(SVector{model.ni[i],T}([3., 3., 0, 1.])) for i=1:p]
R = [Diagonal(zeros(SVector{model.mi[i],T})) for i=1:p] # Quadratic control cost
# Desrired state
xf = [SVector{model.ni[1],T}([0.5, 0., 0, 0]),
      SVector{model.ni[2],T}([0., -0.5, -3.14159/2, 0]),
      SVector{model.ni[3],T}([-0.5, 0., 3.14159, 0]),
      SVector{model.ni[4],T}([0., 0.5, 3.14159/2, 0]),
      ]
# Desired control
uf = [zeros(SVector{model.mi[i],T}) for i=1:p]
# Objectives of the game
game_obj = GameObjective(Q,R,xf,uf,N,model)
radius = 0.06*ones(p)
μ = 50.0*ones(p)
add_collision_cost!(game_obj, radius, μ)

# Define the constraints that each player must respect
game_con = GameConstraintValues(probsize)
# Add collision avoidance
radius = 0.06
add_collision_avoidance!(game_con, radius)
# Add control bounds
u_max = SVector{m,T}([3.14159/2, 3.14159/2, 3.14159/2,  3.14159/2, 0.1, 0.1, 0.1, 0.1])
u_min = SVector{m,T}([-3.14159/2, -3.14159/2, -3.14159/2, -3.14159/2, -0.1, -0.1, -0.1, -0.1])
add_control_bound!(game_con, u_max, u_min)
# # Add state bounds for player 1
x_max = SVector{n,T}([5., 5., 5., 5., 5., 5., 5., 5., 10., 10., 10., 10., 0.2, 0.2, 0.2, 0.2])
x_min = SVector{n,T}([-5., -5., -5., -5., -5., -5., -5., -5., -10., -10., -10., -10., -0.1, -0.1, -0.1, -0.1])
add_state_bound!(game_con, 1, x_max, x_min)
add_state_bound!(game_con, 2, x_max, x_min)
add_state_bound!(game_con, 3, x_max, x_min)
add_state_bound!(game_con, 4, x_max, x_min)

# Define the initial state of the system
x0 = SVector{model.n,T}([
    -0.5, 0., 0.5, 0., 
    0., 0.5, 0., -0.5, 
    0., -3.14159/2, 3.14159, 3.14159/2, 
    0., 0., 0., 0.
    ])

# Define the Options of the solver
opts = Options()
# Define the game problem
prob = GameProblem(N,dt,x0,model,opts,game_obj,game_con)

# Solve the problem
@time newton_solve!(prob)

# Visualize the Results
using Plots
function plot_circles!(plt, xc::AbstractVector, yc::AbstractVector, r::AbstractVector;
                       fillalpha::Real = 0.8,
                       fillcolor = :black,
                       linecolor = :black,
                       linestyle = :solid,
                       linewidth::Real = 2,
                       n::Int = 181)
    @assert length(xc) == length(yc) == length(r) "xc, yc, and r must have same length"
    for i in eachindex(r)
        θ = range(0, 2π; length=n)
        x = xc[i] .+ r[i] .* cos.(θ)
        y = yc[i] .+ r[i] .* sin.(θ)

        # outline
        plot!(plt, x, y; linecolor, linestyle, linewidth, label=nothing)

        # filled disk (optional)
        plot!(plt, [x; x[1]], [y; y[1]];
              seriestype=:shape, fillalpha=fillalpha, fillcolor=fillcolor,
              linecolor=:transparent, label=nothing)
    end
    return plt
end
function plot_traj_with_constraints(prob, r; kwargs...)
    traj = plot!(prob.model, prob.pdtraj.pr)

    xi = [Algames.state(prob.pdtraj.pr[k])[model.pz[1][1]] for k=1:N] 
    yi = [Algames.state(prob.pdtraj.pr[k])[model.pz[1][2]] for k=1:N]
    ri = [0.06 for j=1:N]
    plot_circles!(traj, xi, yi, ri; fillalpha=0.2, fillcolor=:red, linecolor=:red, linestyle=:solid)

    xi = [Algames.state(prob.pdtraj.pr[k])[model.pz[2][1]] for k=1:N] 
    yi = [Algames.state(prob.pdtraj.pr[k])[model.pz[2][2]] for k=1:N]
    ri = [0.06 for j=1:N]
    # plot_circles!(traj, xi, yi, ri; fillalpha=0.5, fillcolor=:blue, linecolor=:blue, linestyle=:solid)

    xlabel!("x"); ylabel!("y"); aspect_ratio=:equal; 
    return traj
end

traj = plot_traj_with_constraints(
    prob, radius;
    fillalpha=0.9, fillcolor=:black, linecolor=:black, linestyle=:solid
)
savefig(traj, "traj.png")

using DelimitedFiles

x1 = [Algames.state(prob.pdtraj.pr[k])[model.pz[1][1]] for k=1:N] 
y1 = [Algames.state(prob.pdtraj.pr[k])[model.pz[1][2]] for k=1:N]
u1_angular = [Algames.control(prob.pdtraj.pr[k])[model.pu[1][1]] for k=1:N]
u1_linear = [Algames.control(prob.pdtraj.pr[k])[model.pu[1][2]] for k=1:N]

x2 = [Algames.state(prob.pdtraj.pr[k])[model.pz[2][1]] for k=1:N] 
y2 = [Algames.state(prob.pdtraj.pr[k])[model.pz[2][2]] for k=1:N]
u2_angular = [Algames.control(prob.pdtraj.pr[k])[model.pu[2][1]] for k=1:N]
u2_linear = [Algames.control(prob.pdtraj.pr[k])[model.pu[2][2]] for k=1:N]

x3 = [Algames.state(prob.pdtraj.pr[k])[model.pz[3][1]] for k=1:N] 
y3 = [Algames.state(prob.pdtraj.pr[k])[model.pz[3][2]] for k=1:N]
u3_angular = [Algames.control(prob.pdtraj.pr[k])[model.pu[3][1]] for k=1:N]
u3_linear = [Algames.control(prob.pdtraj.pr[k])[model.pu[3][2]] for k=1:N]

x4 = [Algames.state(prob.pdtraj.pr[k])[model.pz[4][1]] for k=1:N] 
y4 = [Algames.state(prob.pdtraj.pr[k])[model.pz[4][2]] for k=1:N]
u4_angular = [Algames.control(prob.pdtraj.pr[k])[model.pu[4][1]] for k=1:N]
u4_linear = [Algames.control(prob.pdtraj.pr[k])[model.pu[4][2]] for k=1:N]

X = vcat(x1', y1', u1_angular', u1_linear', x2', y2', u2_angular', u2_linear',
        x3', y3', u3_angular', u3_linear', x4', y4', u4_angular', u4_linear')
writedlm("Algame_debris.csv", X, ',')
