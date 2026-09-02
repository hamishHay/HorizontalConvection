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

    d = sort([parse(Int, split(split(file, "s")[end], ".")[1]) for file in files])

    return d[end]
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
        @printf("Loading iteration %03d from ./data/%s/%03d/snaps2D/snaps2D_s%d.h5\n", itr, suite, N, dfile_num)
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
            qn = data["tasks/heat flux interface"][1, :, itr:itr2]

            d["U"]=U 
            d["V"]=V
            d["f"]=f
            d["T"]=T
            d["ζ"]=vort
            d["x"]=x
            d["z"]=z
            d["t"]=t
            d["qn"]=qn
        end

        dfile_num = latest_snapshot_file(suite, N, "snaps")
        println(dfile_num)
        found = false
        for i in dfile_num:-1:1
            sim_file = @sprintf("./data/%s/%03d/snaps/snaps_s%d.h5", suite, N, i)
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
                    fx = data["tasks/f x"][1, :, itr:itr2]
                    
                    d["qt"] = F
                    d["fx"] = fx

                    found = true
                end 
            end
            if found 
                break
            end
        end
        if !found 
            println("Can't fight where the 2D snapshot should be")
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
    qn = data["qn"][:,1]
    fx = data["fx"][:,1]

    # qt ./= mean(qt)
    ζm = maximum( abs.(extrema(ζ)) )
    
    # -------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------

    fig = Figure( size = (1500 + (Lx[N+1] - 5)/5 * 500, 600) )

    g1 = fig[1:3, 1] = GridLayout()
    g2 = fig[1:3, 2] = GridLayout()

    axTop = Axis(g1[1,1], ylabel = "Heat flux", xticksvisible=false, xticklabelsvisible=false, xminorgridvisible=true, xminorticks=IntervalsBetween(4))
    axMid = Axis(g1[2,1], ylabel = "Ice thickness", xticksvisible=false, xticklabelsvisible=false, xminorgridvisible=true, xminorticks=IntervalsBetween(4))

    ax1 = Axis(g1[3:4, 1], aspect = DataAspect(), ylabel = L"$z$", xticksvisible=false, xticklabelsvisible=false)

    ax2 = Axis(g1[5:6, 1], aspect = DataAspect(), xlabel = L"$x$", ylabel = L"$z$", xminorticksvisible=true, xminorticks=IntervalsBetween(4))

    ax_lke = Axis(g2[1,1], ylabel="Liquid KE", xticklabelsvisible=false, yscale=log10)
    ax_ratio = Axis(g2[2,1], ylabel="Ice--Liquid\nKE ratio", xticklabelsvisible=false, yscale=log10)
    ax_lvol = Axis(g2[3,1], ylabel="Liquid volume")
    ax_flux = Axis(g2[4,1], ylabel="Average surface flux", xlabel="time")


    linkxaxes!(axTop, ax1)
    linkxaxes!(ax1, ax2)

    # # -------------------------------------------------------------------
    # # Temperature plot
    # # -------------------------------------------------------------------

    Tmax = maximum(T[.!isnan.(T)])
    Tmin = minimum(T[.!isnan.(T)])

    lines!(axTop, x, qt; label="surface")
    lines!(axTop, x, qn; label="interface")
    Legend(g1[1, 2], axTop, tellwidth=false, margin=(60, 0, 0, 0))
    lines!(axMid, x, fx)

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
    cb2=Colorbar(g1[4, 2], colormap=reverse(ColorSchemes.ice), colorrange=(0.0, 0.2), ticks=[0.0, 0.1, 0.2], valign=:bottom, height = Relative(0.7) )
    cb =Colorbar(g1[3, 2], colormap=ColorSchemes.seaborn_rocket_gradient, colorrange=(0.2, Tmax), valign=:bottom, 
                 ticks=0.2:round((Tmax-0.2)/3, digits=2):round(Tmax,digits=1))
    Colorbar(g1[5:6, 2], hm2, label = "Vorticity", height = Relative(0.7) )

    cb.alignmode = Mixed(top = 4)
    cb2.alignmode = Mixed(bottom = 4)
    cbar_label = Label(g1[4:3, 2, Right()], "Temperature θ", rotation = pi/2, padding=(25, 0, 0, 0))


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

    ylims!(axTop, 0.0, min(minimum(qt), minimum(qn))*1.1)
    # ylims!(axTop, 0.0, *1.1)
    length(avg_heat_flux) > 1 ? start=2 : start=1
    flux_lim = vcat(avg_heat_flux[start:end]...)
    lke_lim = vcat(KE_liq...)
    
    
    ylims!(ax_flux, mean(flux_lim) - 5std(flux_lim), mean(flux_lim) + 5std(flux_lim))
    # ke_lower = mean(lke_lim) - 3std(lke_lim)
    # ke_lower = ke_lower < 0 ? 0.9minimum(lke_lim) : ke_lower 
    # ylims!(ax_lke, ke_lower, mean(lke_lim) + 3std(lke_lim))

    xlims!(ax1, 0, Lx[N+1])
    xlims!(ax2, 0, Lx[N+1])
    xlims!(axTop, 0, Lx[N+1])
    xlims!(axMid, 0, Lx[N+1])


    colsize!(fig.layout, 2, 450)

    # ax2.alignmode = Mixed(bottom = 0)
    
    
    save_name = @sprintf("./plots/%s_%03d_s%02d_%04d.png", suite, N, s, itr)
    save( save_name, fig, px_per_unit=4)

    return data, fig
end

function get_s_itrs(suite, N; s=nothing)
    snapdir = @sprintf("./data/%s/%03d/%s", suite, N,"snaps2D")

    files = sort(glob("snaps2D_s*.h5", snapdir))


    itrs = [1:length(h5open(file)["scales/iteration/"][1:end-1]) for file in  files]
    snums = [parse(Int, split(split(file, "s")[end], ".")[1]) for file in files]
    ss = [snums[i]*ones(Int, length(itrs[i])) for i in eachindex(snums)]

    if !isnothing(s)
        itrs = itrs[snums .>= s]
        ss    = ss[snums .>= s]
    end

    return vcat(itrs...), vcat(ss...)
end


function plot_animation(suite, N, itrs;
                     filename = "./plots/$(suite)_$(N).mp4",
                     framerate = 10,
                     vort_lim = nothing,
                     temp_lim = nothing, 
                     s1 = nothing,
                     s2 = nothing)

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

    s_itrs, ss = get_s_itrs(suite, N; s=s1)

    if last(itrs) > last(s_itrs)
        error(@sprintf("Selected too many iterations! Max is %d", last(s_itrs)))
    end

    

    itr0 = last(itrs)
    data = get_snapshot(suite, N, s_itrs[last(itrs)]; dfile_num=ss[last(itrs)])

    x, z = data["x"], data["z"]

    T = dropdims(data["T"], dims=3)
    V = dropdims(data["V"], dims=3)
    U = dropdims(data["U"], dims=3)
    f = dropdims(data["f"], dims=3)
    ζ = dropdims(data["ζ"], dims=3)

    snap_time = data["t"][1]

    qt = data["qt"][:, 1]
    # qt ./= mean(qt)

    # Use fixed colour limits over the entire animation.
    # This prevents the colours from changing between frames.
    Tmax = maximum(T[.!isnan.(T)])
    Tmin = minimum(T[.!isnan.(T)])

    if isnothing(vort_lim)
        m = maximum(abs.(extrema(ζ)))
        vort_lim = (-0.1m, 0.1m)
    end

    if isnothing(temp_lim)
        m = maximum(abs.(extrema(T)))
        temp_lim = (0.0, m)
    end

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

    hm1 = heatmap!( ax1, x, z, T_plot, colormap = cmap, colorrange = temp_lim, rasterize = true)

    contour!( ax1, x, z, T_plot; levels = 0.2:(Tmax - 0.2)/7:Tmax, color = (:black, 0.75), linewidth = 0.3)

    # f = 0.5 contour
    contour!( ax1, x, z, f_plot; levels = [0.5], color = (:white, 0.75), linewidth = 0.5)

    hm2 = heatmap!( ax2, x, z, ζ_plot; colormap = :coolwarm, rasterize = true, colorrange = vort_lim)

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
    scat_ker   = nothing
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
            scat_ker = Observable([abs.(KE_ice[i][snap_iter]) ./ KE_liq[i][snap_iter]])
            scat_vol   = Observable([vol_liq[i][snap_iter]] ./ Lx[N+1])
            scat_flux   = Observable([avg_heat_flux[i][snap_iter]])
            scatter!( ax_lke, scat_time, scat_ke, color = (:red, 0.7) )

            scatter!( ax_ratio, scat_time, scat_ker, color = (:red, 0.7) )

            scatter!( ax_lvol, scat_time,  scat_vol, color = (:red, 0.7) )

            scatter!( ax_flux, scat_time, scat_flux, color = (:red, 0.7) )
        end
    end

    # -------------------------------------------------------------------
    # Fixed axis limits
    # -------------------------------------------------------------------

    ylims!(axTop, 0.0,  minimum(qt) * 1.1)
    
    xlims!(ax1, 0, Lx[N+1])
    xlims!(ax2, 0, Lx[N+1])
    xlims!(axTop, 0, Lx[N+1])

    rowsize!(g1, 1, 50.0)
    rowgap!(g1, 1, 10.0)
    rowgap!(g1, 3, 10.0)
    rowgap!(g1, 2, 0)
    colsize!(fig.layout, 2, 500)

    
    
    record(fig, filename, itrs; framerate = framerate) do i

        # ---------------------------------------------------------------
        # Read snapshot for this iteration
        # ---------------------------------------------------------------

        itr = s_itrs[i]
        s   = ss[i]

        data = get_snapshot(suite, N, itr; dfile_num=s, no_series=true)

        T_new = dropdims(data["T"], dims=3)
        V_new = dropdims(data["V"], dims=3)
        U_new = dropdims(data["U"], dims=3)
        f_new = dropdims(data["f"], dims=3)
        ζ_new = dropdims(data["ζ"], dims=3)
        qt_new = data["qt"][:, 1]

        snap_time = data["t"][1]

        T_obs[] = T_new
        V_obs[] = V_new
        U_obs[] = U_new ./ maximum(sqrt.(U_new.^2 .+ V_new.^2))
        V_obs[] = V_new ./ maximum(sqrt.(U_new.^2 .+ V_new.^2))
        f_obs[] = f_new
        ζ_obs[] = ζ_new
        qt_obs[] = qt_new

        snap_iter = 1
        for j in eachindex(times)
            if snap_time >= extrema(times[j])[1] &&  snap_time <= extrema(times[j])[2]
                snap_iter = findmin(abs.(times[j] .- snap_time))[2]            
            end
        end

        scat_time[] = [times[s][snap_iter]]
        scat_ke[] = [KE_liq[s][snap_iter]]
        scat_ker[] = [abs.(KE_ice[s][snap_iter]) ./ KE_liq[s][snap_iter]]
        scat_vol[] = [vol_liq[s][snap_iter] ./ Lx[N+1] ]
        scat_flux[] = [avg_heat_flux[s][snap_iter] ]
        qt_obs[] = qt_new

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
