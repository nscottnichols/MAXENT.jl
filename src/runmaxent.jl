using ArgParse
using Random
using JLD
using NPZ
using UUIDs

include("./MAXENT.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--temperature", "-T"
            help = "Temperature of system."
            arg_type = Float64
            default = 0.0
        "--number_of_iterations", "-N"
            help = "Number of generations before algorithm quits."
            arg_type = Int64
            default = 1000
        "--regularization_constant"
            help = "Regularization constant."
            arg_type = Float64
            default = 0.0
        "--save_file_dir"
            help = "Directory to save results in."
            arg_type = String
            default = "./maxentresults"
        "qmc_data"
            help = "*.npz or *.jld file containing quantum Monte Carlo data with columns: IMAGINARY_TIME, INTERMEDIATE_SCATTERING_FUNCTION, ERROR"
            arg_type = String
            required = true
        "initial_dsf"
            help = "*.npz or *.jld file containing initial guess for dynamic structure factor and frequency with columns: FREQUENCY, DSF"
            arg_type = String
            required = true
        "default_dsf"
            help = "*.npz or *.jld file containing default dynamic structure factor with columns: FREQUENCY,DSF"
            arg_type = String
            required = true
    end

    return parse_args(s)
end

function main()
    start = time();
    parsed_args = parse_commandline()

    # create data directory
    try
        mkpath(parsed_args["save_file_dir"]);
    catch
        nothing
    end
    save_dir = parsed_args["save_file_dir"];

    _ext = splitext(parsed_args["qmc_data"])[2];
    if _ext == ".npz"
        qmcdata=NPZ.npzread(parsed_args["qmc_data"]);
    elseif _ext == ".jld"
        qmcdata = load(parsed_args["qmc_data"]);
    else
        throw(AssertionError("qmc_data must be *.jld or *.npz"));
    end

    imaginary_time = qmcdata["tau"];
    isf = qmcdata["isf"];
    isf_error = qmcdata["error"];

    _ext = splitext(parsed_args["initial_dsf"])[2];
    if _ext == ".npz"
        dsfdata=NPZ.npzread(parsed_args["initial_dsf"]);
    elseif _ext == ".jld"
        dsfdata = load(parsed_args["initial_dsf"]);
    else
        throw(AssertionError("initial_dsf must be *.jld or *.npz"));
    end
    dsf = dsfdata["dsf"];
    frequency_bins = dsfdata["frequency"];

    _ext = splitext(parsed_args["default_dsf"])[2];
    if _ext == ".npz"
        defaultdsfdata=NPZ.npzread(parsed_args["default_dsf"]);
    elseif _ext == ".jld"
        defaultdsfdata = load(parsed_args["default_dsf"]);
    else
        throw(AssertionError("default_dsf must be *.jld or *.npz"));
    end
    default_dsf = defaultdsfdata["dsf"];

    u4 = uuid4();
    results, quality_of_fit = MAXENT.maxent(dsf,default_dsf,isf,isf_error,
                           frequency_bins,imaginary_time,
                           temperature = parsed_args["temperature"],
                           regularization_constant=parsed_args["regularization_constant"],
                           number_of_iterations=parsed_args["number_of_iterations"])
    elapsed = time() - start;
    regularization_constant=parsed_args["regularization_constant"];
    filename = "$(save_dir)/maxent_results_$(regularization_constant)_$u4.jld";
    println("Saving results to $filename");
    save(filename,
         "u4",u4,
         "results",results,
         "frequency",frequency_bins,
         "elapsed_time",elapsed,
         "quality_of_fit",quality_of_fit,
         "regularization_constant",regularization_constant);
    filename = "$(save_dir)/maxent_params_$(regularization_constant)_$u4.jld";
    println("Saving parameters to $filename");
    save(filename,
         "u4",u4,
         "imaginary_time",imaginary_time,
         "isf",isf,
         "isf_error",isf_error,
         "frequency",frequency_bins,
         "initial_dsf",dsf,
         "default_dsf",default_dsf,
         "number_of_iterations",parsed_args["number_of_iterations"],
         "temperature",parsed_args["temperature"],
         "regularization_constant",parsed_args["regularization_constant"])
    nothing
end

main()
