using ColorSchemes
using HDF5
using CairoMakie
using CSV
using DataFrames
using Printf
using Statistics
using Glob

function latest_snapshot_file(suite, N, fname)
    snapdir = @sprintf("./data/%s/%03d/%s", suite, N,fname)

    files = sort(glob(fname*"_s*.h5", snapdir))
    
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

function get_time_series(suite, N; itr1=nothing, itr2=nothing, d=Dict(), sim_iteration=false)
    if isnothing(itr1)
        itr1 = 1
    end

    dfile_num = latest_snapshot_file(suite, N, "snaps")
    sim_file = @sprintf("./data/%s/%03d/snaps/snaps_s%d.h5", suite, N, dfile_num)

    snapshot_pattern = @sprintf("./data/%s/%03d/snaps/snaps*.h5", suite,N)

    files = sort(glob(snapshot_pattern))

    KE_liq = []
    KE_ice = []
    vol_liq = []
    avg_heat_flux = []
    heat_flux_x = []
    times = []
    for sim_file in files
        h5open(sim_file, "r") do data
            println("$sim_file found")
            max_iter = length(data["scales/sim_time"][:])
            if isnothing(itr2)
                itr2=max_iter
            end

            push!(KE_liq, data["tasks/KE liq"][1,1,itr1:itr2])
            push!(KE_ice, data["tasks/KE ice"][1,1,itr1:itr2])
            push!(vol_liq, data["tasks/vol liq"][1,1,itr1:itr2])
            push!(times, data["scales/sim_time"][itr1:itr2])
            # end
            # println(size(mean(data["tasks/heat flux top x"][1, :, :], dims=1)))
            push!(avg_heat_flux, mean(data["tasks/heat flux top x"][1, :, itr1:itr2], dims=1)[1,:])
            push!(heat_flux_x, data["tasks/heat flux top x"][1, :, itr1:itr2])
        end
    end
    d["KE liq"] = KE_liq
    d["KE ice"] = KE_ice
    d["vol liq"] = vol_liq
    d["times"] = times
    d["avg heat flux"] = avg_heat_flux
    d["heat flux x"] = heat_flux_x

    return d
end

function get_snapshot(suite, N, itr; itr2=nothing, no_series = false, dfile_num=nothing)

        if isnothing(dfile_num)
            dfile_num = latest_snapshot_file(suite, N, "snaps2D")
        end

        d = Dict()
        sim_file = @sprintf("./data/%s/%03d/snaps2D/snaps2D_s%d.h5", suite, N, dfile_num)
        snapshot_itr = 1
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
            snapshot_itr = data["scales/iteration"][itr:itr2]
            @printf("Reading from time %1.4f--%1.4f, sim iterations %d--%d\n", t[1], t[end], snapshot_itr[1], snapshot_itr[end])

            U = data["tasks/velocity"][:, :, 1, itr:itr2]
            V = data["tasks/velocity"][:, :, 2, itr:itr2]
            vort = data["tasks/vorticity"][:,:,itr:itr2]
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

        dfile_num = latest_snapshot_file(suite, N, "snaps")
        for i in dfile_num:-1:1
            sim_file = @sprintf("./data/%s/%03d/snaps/snaps_s%d.h5", suite, N, i)
            found = false
            h5open(sim_file, "r") do data
                itr>0 ? nothing : error("Iteration should be greater than 0")
                println("$sim_file found")

                max_iter = length(data["scales/sim_time"][:])

                scales = data["scales"]
                sim_itrs = data["scales/iteration"][:]
                snapshot_index = findfirst(sim_itrs .== snapshot_itr)  
                
                
                if !isnothing(snapshot_index)
                    itr = itr2 = snapshot_index
                    x = find_scale(scales, "x")
                    t = data["scales/sim_time"][itr:itr2]
                    sim_itr = data["scales/iteration"][itr:itr2]
                    @printf("Reading from time %1.4f--%1.4f, sim iterations %d--%d, save index %d--%d\n", t[1], t[end], sim_itr[1], sim_itr[end], itr, itr2)
                    F = data["tasks/heat flux top x"][1, :, itr:itr2]
                    
                    d["qt"] = F

                    found = true
                end 
            end
            if found 
                break
            end
        end

        if !no_series
            snapshot_pattern = @sprintf("./data/%s/%03d/snaps/snaps*.h5", suite,N)

            files = sort(glob(snapshot_pattern))

            KE_liq = []
            KE_ice = []
            vol_liq = []
            avg_heat_flux = []
            times = []
            for sim_file in files
                h5open(sim_file, "r") do data
                    println("$sim_file found")

                    push!(KE_liq, data["tasks/KE liq"][1,1,:])
                    push!(KE_ice, data["tasks/KE ice"][1,1,:])
                    push!(vol_liq, data["tasks/vol liq"][1,1,:])
                    # if length(data["tasks/KE liq"][1,1,:]) != length(data["scales/sim_time"][:]) 
                    #     push!(times, data["scales/sim_time"][1:end-1])
                    # else
                    push!(times, data["scales/sim_time"][:])
                    # end
                    # println(size(mean(data["tasks/heat flux top x"][1, :, :], dims=1)))
                    push!(avg_heat_flux, mean(data["tasks/heat flux top x"][1, :, :], dims=1)[1,:])
                end
            end
            d["KE liq"] = KE_liq
            d["KE ice"] = KE_ice
            d["vol liq"] = vol_liq
            d["times"] = times
            d["avg heat flux"] = avg_heat_flux
        end


        return d
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

function plot_latest(suite, N, itr; s=nothing)

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

    data = get_snapshot(suite, N, itr; dfile_num=s)
    if isnothing(s)
        s = latest_snapshot_file(suite, N, "snaps2D")
    end

    x, z= data["x"], data["z"]
    T = dropdims(data["T"], dims=3)
    V = dropdims(data["V"], dims=3)
    U = dropdims(data["U"], dims=3)
    f = dropdims(data["f"], dims=3)
    ζ = dropdims(data["ζ"], dims=3)
    snap_time = data["t"][1]
    qt = data["qt"][:,1]

    qt ./= mean(qt)
    ζm = maximum( abs.(extrema(ζ)) )
    
    # -------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------

    fig = Figure( size = (1500 + (Lx[N+1] - 5)/5 * 500, 500) )

    g1 = fig[1:3, 1] = GridLayout()
    g2 = fig[1:3, 2] = GridLayout()

    axTop = Axis(g1[1,1], ylabel = "Surface\nheat flux", xticksvisible=false, xticklabelsvisible=false, xminorgridvisible=true, xminorticks=IntervalsBetween(4))

    ax1 = Axis(g1[2:3, 1], aspect = DataAspect(), ylabel = L"$z$", xticksvisible=false, xticklabelsvisible=false)

    ax2 = Axis(g1[4:5, 1], aspect = DataAspect(), xlabel = L"$x$", ylabel = L"$z$", xminorticksvisible=true, xminorticks=IntervalsBetween(4))

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
    hm1 = heatmap!(ax1, x, z, T', colormap = cmap, colorrange=(0.0, Tmax), rasterize=true)
    contour!(ax1,x, z, T'; levels = 0.2:(Tmax-0.2)/7:Tmax, color = (:black, 0.75), linewidth = 0.3)

    # f = 0.5 contour
    contour!(ax1,x, z, f'; levels = [0.5], color = (:white, 0.75), linewidth = 0.5)

    hm2 = heatmap!(ax2, x, z, ζ'; colormap = :coolwarm, rasterize=true, colorrange=(-ζm*0.1, ζm*0.1))
    contour!(ax2,x, z, f'; levels = [0.5], color = (:black, 0.75), linewidth = 0.75)

    skipz = 64
    skipx = 16
    
    mag = maximum(sqrt.(U.^2 .+ V.^2))
    U ./= mag
    V ./= mag
    arrows2d!(ax2, x[1:skipx:end], z[1:skipz:end], U[1:skipz:end, 1:skipx:end]', V[1:skipz:end, 1:skipx:end]', lengthscale=0.1)

    # # -------------------------------------------------------------------
    # # Colorbars
    # # -------------------------------------------------------------------

    # Colorbar(g1[1, 2], hm1, label = "Temperature θ", height = Relative(0.7), ticks=0.0:0.1:round(Tmax, digits=1) )
    cb2=Colorbar(g1[3, 2], colormap=reverse(ColorSchemes.ice), colorrange=(0.0, 0.2), ticks=[0.0, 0.1, 0.2], valign=:bottom, height = Relative(0.7) )
    cb =Colorbar(g1[2, 2], colormap=ColorSchemes.seaborn_rocket_gradient, colorrange=(0.2, Tmax), valign=:bottom, 
                 ticks=0.2:round((Tmax-0.2)/3, digits=2):round(Tmax,digits=1))
    Colorbar(g1[4:5, 2], hm2, label = "Vorticity", height = Relative(0.7) )

    cb.alignmode = Mixed(top = 4)
    cb2.alignmode = Mixed(bottom = 4)
    cbar_label = Label(g1[3:2, 2, Right()], "Temperature θ", rotation = pi/2, padding=(25, 0, 0, 0))


    KE_liq = data["KE liq"]
    KE_ice = data["KE ice"]
    vol_liq = data["vol liq"]
    avg_heat_flux = data["avg heat flux"]
    times = data["times"]

    for i in eachindex(KE_liq)
        
        lines!(ax_lke, times[i], KE_liq[i])
        lines!(ax_ratio, times[i], abs.(KE_ice[i])./KE_liq[i])
        lines!(ax_lvol, times[i], vol_liq[i] ./ Lx[N+1])
        lines!(ax_flux, times[i], avg_heat_flux[i])

        snap_iter = 1
        if snap_time >= extrema(times[i])[1] && snap_time <= extrema(times[i])[2]
            snap_iter = findmin(abs.(times[i] .- snap_time))[2] 
            scatter!(ax_lke, times[i][snap_iter], KE_liq[i][snap_iter], color=(:red, 0.7))
            scatter!(ax_ratio, times[i][snap_iter], abs.(KE_ice[i][snap_iter])./KE_liq[i][snap_iter], color=(:red, 0.7))
            scatter!(ax_lvol, times[i][snap_iter], vol_liq[i][snap_iter] ./ Lx[N+1], color=(:red, 0.7))
            scatter!(ax_flux, times[i][snap_iter], avg_heat_flux[i][snap_iter], color=(:red, 0.7))
        end
    end

    ylims!(axTop, 0.0, maximum(qt)*1.1)
    ylims!(ax_flux, -1.5, -0.5)

    xlims!(ax1, 0, Lx[N+1])
    xlims!(ax2, 0, Lx[N+1])
    xlims!(axTop, 0, Lx[N+1])

    
    rowsize!(g1, 1, 50.0)
    rowgap!(g1, 1, 10.0)
    rowgap!(g1, 3, 10.0)
    rowgap!(g1, 2, 0)
    colsize!(fig.layout, 2, 500)
    
    
    save_name = @sprintf("./plots/%s_%03d_s%02d_%04d.png", suite, N, s, itr)
    save( save_name, fig, px_per_unit=4)

    return data, fig
end

function plot_animation(suite, N, itrs;
                     filename = "./plots/$(suite)_$(N).mp4",
                     framerate = 10)

    # -------------------------------------------------------------------
    # Read parameters
    # -------------------------------------------------------------------

    parameter_file = "./parameters/parameters-$(suite).csv"
    df = CSV.read(parameter_file, DataFrame)

    Tm = df.Tm
    Ra = df.Ra
    Lx = df.Lx

    # -------------------------------------------------------------------
    # Get first snapshot -- used to construct the figure
    # -------------------------------------------------------------------

    itr0 = first(itrs)
    data = get_snapshot(suite, N, itr0)

    x, z = data["x"], data["z"]

    T = dropdims(data["T"], dims=3)
    V = dropdims(data["V"], dims=3)
    U = dropdims(data["U"], dims=3)
    f = dropdims(data["f"], dims=3)
    ζ = dropdims(data["ζ"], dims=3)

    snap_time = data["t"][1]

    qt = copy(data["qt"][:, 1])
    qt ./= mean(qt)

    # Use fixed colour limits over the entire animation.
    # This prevents the colours from changing between frames.
    Tmax = maximum(T[.!isnan.(T)])
    Tmin = minimum(T[.!isnan.(T)])

    ζm = maximum(abs.(extrema(ζ)))

    # -------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------

    fig = Figure(size = (1500 + (Lx[N+1] - 5) / 5 * 500, 500) )

    g1 = fig[1:3, 1] = GridLayout()
    g2 = fig[1:3, 2] = GridLayout()

    axTop = Axis( g1[1, 1], ylabel = "Surface\nheat flux", 
                  xticksvisible = false, xticklabelsvisible = false, 
                  xminorgridvisible = true, xminorticks = IntervalsBetween(4) )

    ax1 = Axis( g1[2:3, 1], aspect = DataAspect(), ylabel = L"$z$", xticksvisible = false, xticklabelsvisible = false)

    ax2 = Axis( g1[4:5, 1], aspect = DataAspect(), xlabel = L"$x$", ylabel = L"$z$", xminorticksvisible = true, xminorticks = IntervalsBetween(4))

    ax_lke = Axis(g2[1, 1], ylabel = "Liquid KE", xticklabelsvisible = false)
    ax_ratio = Axis(g2[1, 2], ylabel = "Ice--Liquid\nKE ratio", xticklabelsvisible = false)
    ax_lvol = Axis(g2[2, 1], ylabel = "Liquid volume")
    ax_flux = Axis(g2[2, 2], ylabel = "Average surface flux")

    linkxaxes!(axTop, ax1)
    linkxaxes!(ax1, ax2)

    # -------------------------------------------------------------------
    # Observables
    # -------------------------------------------------------------------

    T_obs  = Observable(T)
    ζ_obs  = Observable(ζ)
    f_obs  = Observable(f)
    U_obs  = Observable(U)
    V_obs  = Observable(V)
    qt_obs = Observable(qt)


    # Transpose the matrices through lifts so that they update correctly.
    T_plot = @lift $T_obs'
    ζ_plot = @lift $ζ_obs'
    f_plot = @lift $f_obs'

    # -------------------------------------------------------------------
    # Temperature / heat-flux plots
    # -------------------------------------------------------------------

    
    qt_line = lines!(axTop, x, qt_obs; color = :black)

    cmap = join_colormaps(
        reverse(ColorSchemes.ice),
        ColorSchemes.seaborn_rocket_gradient,
        0.2;
        xmin = 0,
        xmax = Tmax)

    hm1 = heatmap!( ax1, x, z, T_plot, colormap = cmap, colorrange = (0.0, Tmax), rasterize = true)

    contour!( ax1, x, z, T_plot; levels = 0.2:(Tmax - 0.2)/7:Tmax, color = (:black, 0.75), linewidth = 0.3)

    # f = 0.5 contour
    contour!( ax1, x, z, f_plot; levels = [0.5], color = (:white, 0.75), linewidth = 0.5)

    hm2 = heatmap!( ax2, x, z, ζ_plot; colormap = :coolwarm, rasterize = true, colorrange = (-ζm * 0.1, ζm * 0.1))

    contour!( ax2, x, z, f_plot; levels = [0.5], color = (:black, 0.75), linewidth = 0.75)

    # -------------------------------------------------------------------
    # Velocity arrows
    # -------------------------------------------------------------------

    skipz = 64
    skipx = 16

    mag = maximum(sqrt.(U.^2 .+ V.^2))

    U ./= mag
    V ./= mag

    U_arrow = @lift $U_obs[1:skipz:end, 1:skipx:end]'
    V_arrow = @lift $V_obs[1:skipz:end, 1:skipx:end]'

    arrows2d!( ax2, x[1:skipx:end], z[1:skipz:end], U_arrow, V_arrow, lengthscale = 0.1)

    # -------------------------------------------------------------------
    # Colorbars
    # -------------------------------------------------------------------

    cb2 = Colorbar( g1[3, 2], colormap = reverse(ColorSchemes.ice), colorrange = (0.0, 0.2), ticks = [0.0, 0.1, 0.2], valign = :bottom, height = Relative(0.7) )

    cb = Colorbar( g1[2, 2], colormap = ColorSchemes.seaborn_rocket_gradient, colorrange = (0.2, Tmax), valign = :bottom, ticks = 0.2:round((Tmax - 0.2) / 3, digits=2):round(Tmax, digits=1) )

    Colorbar( g1[4:5, 2], hm2, label = "Vorticity", height = Relative(0.7) )

    cb.alignmode = Mixed(top = 4)
    cb2.alignmode = Mixed(bottom = 4)

    Label( g1[3:2, 2, Right()], "Temperature θ", rotation = pi / 2, padding = (25, 0, 0, 0) )

    # -------------------------------------------------------------------
    # Time-series data
    # -------------------------------------------------------------------
    scat_time = nothing
    scat_ke   = nothing
    scat_vol   = nothing 
    scat_flux   = nothing 
    KE_liq = data["KE liq"]
    KE_ice = data["KE ice"]
    vol_liq = data["vol liq"]
    avg_heat_flux = data["avg heat flux"]
    times = data["times"]

    for i in eachindex(KE_liq)

        lines!(ax_lke, times[i], KE_liq[i])
        lines!(ax_ratio, times[i], abs.(KE_ice[i]) ./ KE_liq[i])
        lines!(ax_lvol, times[i], vol_liq[i] ./ Lx[N+1])
        lines!(ax_flux, times[i], avg_heat_flux[i])

        if snap_time >= extrema(times[i])[1] &&
           snap_time <= extrema(times[i])[2]

            snap_iter = findmin(abs.(times[i] .- snap_time))[2]

            scat_time = Observable([times[i][snap_iter]])
            scat_ke   = Observable([KE_liq[i][snap_iter]])
            scat_vol   = Observable([vol_liq[i][snap_iter]] ./ Lx[N+1])
            scat_flux   = Observable([avg_heat_flux[i][snap_iter]])
            scatter!( ax_lke, scat_time, scat_ke, color = (:red, 0.7) )

            scatter!( ax_ratio, times[i][snap_iter], abs.(KE_ice[i][snap_iter]) ./ KE_liq[i][snap_iter], color = (:red, 0.7) )

            scatter!( ax_lvol, scat_time,  scat_vol, color = (:red, 0.7) )

            scatter!( ax_flux, scat_time, scat_flux, color = (:red, 0.7) )
        end
    end

    # -------------------------------------------------------------------
    # Fixed axis limits
    # -------------------------------------------------------------------

    ylims!(axTop, 0.0, maximum(qt) * 1.1)

    xlims!(ax1, 0, Lx[N+1])
    xlims!(ax2, 0, Lx[N+1])
    xlims!(axTop, 0, Lx[N+1])

    rowsize!(g1, 1, 50.0)
    rowgap!(g1, 1, 10.0)
    rowgap!(g1, 3, 10.0)
    rowgap!(g1, 2, 0)
    colsize!(fig.layout, 2, 500)

    # -------------------------------------------------------------------
    # Animation
    # -------------------------------------------------------------------

    record(
        fig,
        filename,
        itrs;
        framerate = framerate
    ) do itr

        # ---------------------------------------------------------------
        # Read snapshot for this iteration
        # ---------------------------------------------------------------

        data = get_snapshot(suite, N, itr; no_series=true)
        # data2 = get_time_series(suite, N; itr1=itr, itr2=itr)

        T_new = dropdims(data["T"], dims=3)
        V_new = dropdims(data["V"], dims=3)
        U_new = dropdims(data["U"], dims=3)
        f_new = dropdims(data["f"], dims=3)
        ζ_new = dropdims(data["ζ"], dims=3)

        snap_time = data["t"][1]

        

        # qt_new = copy(data2["heat flux x"][:, 1])
        # qt_new ./= mean(qt_new)

        # qt_obs[] = qt_new

        # ---------------------------------------------------------------
        # Update fields
        # ---------------------------------------------------------------

        T_obs[] = T_new
        V_obs[] = V_new
        U_obs[] = U_new ./ maximum(sqrt.(U_new.^2 .+ V_new.^2))
        V_obs[] = V_new ./ maximum(sqrt.(U_new.^2 .+ V_new.^2))
        f_obs[] = f_new
        ζ_obs[] = ζ_new

        if snap_time >= extrema(times[end])[1] &&
           snap_time <= extrema(times[end])[2]

            snap_iter = findmin(abs.(times[end] .- snap_time))[2]

        end

        scat_time[] = [times[end][snap_iter]]
        scat_ke[] = [KE_liq[end][snap_iter]]
        scat_vol[] = [vol_liq[end][snap_iter] ./ Lx[N+1] ]
        scat_flux[] = [avg_heat_flux[end][snap_iter] ]
        # qt_obs[] = qt_new

        # ---------------------------------------------------------------
        # Update title / frame information
        # ---------------------------------------------------------------

        axTop.title =  @sprintf("Iteration %03d    t = %0.3f", itr, data["t"][1])

        # Make sure Makie processes the Observable updates.
        autolimits!(ax_lke)
        autolimits!(ax_ratio)
        autolimits!(ax_lvol)
        autolimits!(ax_flux)
    end

    return fig
end
