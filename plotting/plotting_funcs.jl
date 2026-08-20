using ColorSchemes
using HDF5
using GLMakie
using CSV
using DataFrames
using Printf
using Statistics
using Glob

function latest_snapshot_file(suite, N)
    snapdir = @sprintf("./data/%s/%03d/snaps2D", suite, N)

    files = sort(glob("snaps2D_s*.h5", snapdir))
    
    isempty(files) && error("No snapshot files found in $snapdir")

    # println(parse(Int, Convertsplit(split(files[end], "s")[end], ".")[1]))

    return parse(Int, split(split(files[end], "s")[end], ".")[1])  
end

function find_scale(scales, name)
    for key in keys(scales)
        obj = scales[key]

        # Ignore non-dataset objects
        if obj isa HDF5.Dataset
            a = attrs(obj)

            if haskey(a, "NAME") && String(a["NAME"]) == name
                return read(obj)
            end
        end
    end

    error("Could not find scale with NAME = \"$name\"")
end

function get_snapshot(suite, N, itr; itr2=nothing)

        dfile_num = latest_snapshot_file(suite, N)

        d = Dict()
        sim_file = @sprintf("./data/%s/%03d/snaps2D/snaps2D_s%d.h5", suite, N, dfile_num)
        h5open(sim_file, "r") do data

            itr>0 ? nothing : error("Iteration should be greater than 0")
            println("$sim_file found")

            max_iter = length(data["scales/sim_time"][:])

            if isnothing(itr2)
                itr2 = itr
            elseif itr2=="end"
                itr2 = max_iter
            end
            itr2 <= max_iter ? println("Maximum iteration is $(max_iter)") : error("Iteration $(itr2) is outside of range (0-$(max_iter))")

            scales = data["scales"]

            x = find_scale(scales, "x")
            z = find_scale(scales, "z")

            t = data["scales/sim_time"][itr:itr2]
            sim_itr = data["scales/iteration"][itr:itr2]
            @printf("Reading from time %1.4f--%1.4f, sim iterations %d--%d", t[1], t[end], sim_itr[1], sim_itr[end])

            # Velocity components
            U = data["tasks/velocity"][:, :, 1, itr:itr2]
            V = data["tasks/velocity"][:, :, 2, itr:itr2]

            vort = data["tasks/vorticity"][:,:,itr:itr2]

            # Other fields
            f = data["tasks/phase"][:, :, itr:itr2]
            T = data["tasks/temperature"][:, :, itr:itr2]

            d["U"]=U 
            d["V"]=V
            d["f"]=f
            d["T"]=T
            d["ζ"]=vort
            d["x"]=x
            d["z"]=z
            d["t"]=t
        end

        sim_file = @sprintf("./data/%s/%03d/snaps/snaps_s%d.h5", suite, N, dfile_num)
        h5open(sim_file, "r") do data

            itr>0 ? nothing : error("Iteration should be greater than 0")
            println("$sim_file found")

            max_iter = length(data["scales/sim_time"][:])

            if isnothing(itr2)
                itr2 = itr
            elseif itr2=="end"
                itr2 = max_iter
            end
            itr2 <= max_iter ? println("Maximum iteration is $(max_iter)") : error("Iteration $(itr2) is outside of range (0-$(max_iter))")

            scales = data["scales"]

            x = find_scale(scales, "x")

            t = data["scales/sim_time"][itr:itr2]
            sim_itr = data["scales/iteration"][itr:itr2]
            @printf("Reading from time %1.4f--%1.4f, sim iterations %d--%d", t[1], t[end], sim_itr[1], sim_itr[end])

            F = data["tasks/heat flux top x"][1, :, itr:itr2]
            
            d["qt"] = F
        end

        snapshot_pattern = @sprintf("./data/%s/%03d/snaps/snaps*.h5", suite,N)

        files = sort(glob(snapshot_pattern))

        KE_liq = []
        KE_ice = []
        vol_liq = []
        avg_heat_flux = []
        times = []
        for sim_file in files
            h5open(sim_file, "r") do data
                push!(KE_liq, data["tasks/KE liq"][1,1,:])
                push!(KE_ice, data["tasks/KE ice"][1,1,:])
                push!(vol_liq, data["tasks/vol liq"][1,1,:])
                push!(times, data["scales/sim_time"][:])
                println(size(mean(data["tasks/heat flux top x"][1, :, :], dims=1)))
                push!(avg_heat_flux, mean(data["tasks/heat flux top x"][1, :, :], dims=1)[1,:])
            end
        end
        d["KE liq"] = KE_liq
        d["KE ice"] = KE_ice
        d["vol liq"] = vol_liq
        d["times"] = KE_liq
        d["avg heat flux"] = avg_heat_flux


        return d
    # catch err
    #     println("$sim_file not found")
    #     return nothing
    # end
end

# function get_snapshot(sim_file, itr1, itr2)
#     return get_snapshot(sim_file, itr1; itr2=itr2)
# end

function join_colormaps(cmap1, cmap2, xmid; xmin=0, xmax=1, n=1024)
    xlen = xmax - xmin 
    cm1_ratio = (xmid - xmin) / xlen 
    cm2_ratio = (xmax - xmid) / xlen 
    
    n1 = round(Int, cm1_ratio * n) - 1
    n2 = n - n1
    c1 = get(cmap1, LinRange(0, 1, n1))
    c2 = get(cmap2, LinRange(0, 1, n2))

    return vcat(c1, c2)
end

function plot_latest(suite, N, itr)

    # # -------------------------------------------------------------------
    # # Read parameters
    # # -------------------------------------------------------------------

    parameter_file = "./parameters/parameters-$(suite).csv"
    df = CSV.read(parameter_file, DataFrame)

    Tm = df.Tm
    Ra = df.Ra
    Lx = df.Lx
    
    # # -------------------------------------------------------------------
    # # Snapshot file
    # # -------------------------------------------------------------------

    data = get_snapshot(suite, N, itr)

    x, z= data["x"], data["z"]
    T = dropdims(data["T"], dims=3)
    V = dropdims(data["V"], dims=3)
    U = dropdims(data["U"], dims=3)
    f = dropdims(data["f"], dims=3)
    ζ = dropdims(data["ζ"], dims=3)
    qt = data["qt"][:,1]

    qt ./= mean(qt)
    
    # -------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------

    fig = Figure( size = (1500, 400) )

    g1 = fig[1:3, 1] = GridLayout()
    g2 = fig[1:3, 2] = GridLayout()

    axTop = Axis(g1[1,1], ylabel = "Surface\nheat flux", xticksvisible=false, xticklabelsvisible=false, xminorgridvisible=true, xminorticks=IntervalsBetween(4))

    ax1 = Axis(g1[2, 1], aspect = DataAspect(), ylabel = L"$z$", xticksvisible=false, xticklabelsvisible=false)

    ax2 = Axis(g1[3, 1], aspect = DataAspect(), xlabel = L"$x$", ylabel = L"$z$", xminorticksvisible=true, xminorticks=IntervalsBetween(4))

    ax_lke = Axis(g2[1,1], ylabel="Liquid KE", xticklabelsvisible=false)
    ax_ratio = Axis(g2[1,2], ylabel="Ice--Liquid\nKE ratio", xticklabelsvisible=false)
    ax_lvol = Axis(g2[2,1], ylabel="Liquid volume")
    ax_flux = Axis(g2[2,2], ylabel="Average surface flux")


    linkxaxes!(axTop, ax1)
    linkxaxes!(ax1, ax2)

    # # -------------------------------------------------------------------
    # # Temperature plot
    # # -------------------------------------------------------------------

    Tmax = maximum(T[.!isnan.(T)])
    Tmin = minimum(T[.!isnan.(T)])

    lines!(axTop, x, qt)

    cmap = join_colormaps(reverse(ColorSchemes.ice), ColorSchemes.seaborn_rocket_gradient, 0.2; xmin=0, xmax=Tmax)
    hm1 = heatmap!(ax1, x, z, T', colormap = cmap, colorrange=(0.0, Tmax))
    contour!(ax1,x, z, T'; levels = 0.2:0.025:round(Tmax, digits=1), color = (:black, 0.75), linewidth = 0.5)

    # f = 0.5 contour
    contour!(ax1,x, z, f'; levels = [0.5], color = (:white, 0.75), linewidth = 0.7)

    hm2 = heatmap!(ax2, x, z, ζ'; colormap = :coolwarm)

    skipz = 64
    skipx = 16
    
    arrows2d!(ax2, x[1:skipx:end], z[1:skipz:end], U[1:skipz:end, 1:skipx:end]', V[1:skipz:end, 1:skipx:end]', lengthscale=5e-4)

    # # -------------------------------------------------------------------
    # # Colorbars
    # # -------------------------------------------------------------------

    # Colorbar(g1[1, 2], hm1, label = "Temperature θ", height = Relative(0.7), ticks=0.0:0.1:round(Tmax, digits=1) )
    Colorbar(g1[2, 2], hm1, label = "Temperature θ", height = Relative(0.7), ticks=0.0:0.1:round(Tmax, digits=1) )
    Colorbar(g1[3, 2], hm2, label = "Vorticity", height = Relative(0.7) )

    # save(
    #     save_name,
    #     fig;
    #     px_per_unit = 1
    # )

    KE_liq = data["KE liq"]
    KE_ice = data["KE ice"]
    vol_liq = data["vol liq"]
    avg_heat_flux = data["avg heat flux"]
    times = data["times"]
    
    for i in eachindex(KE_liq)
        # println(KE[i])
        lines!(ax_lke, times[i], KE_liq[i])
        lines!(ax_ratio, times[i], KE_ice[i]./KE_liq[i])
        lines!(ax_lvol, times[i], vol_liq[i] ./ Lx[N+1])
        lines!(ax_flux, times[i], avg_heat_flux[i])
        # lines!(ax_lke, times[i], KE_ice[i])
        println(KE_ice[i]./KE_liq[i])
    end

    
    # println("Saved: $save_name")

    ylims!(axTop, 0.0, 2.0)

    rowsize!(g1, 1, 50.0)
    rowgap!(g1, 1, 10.0)
    rowgap!(g1, 2, 10.0)

    save( @sprintf("./plots/%s_%03d_%04.png", suite, N, itr), px_per_inch=4)

    return data, fig
end