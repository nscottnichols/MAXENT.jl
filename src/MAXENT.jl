module MAXENT
using Random
using Statistics
using LinearAlgebra
using Optim

include("./MAXENT_model_isf.jl")

export maxent

function entropy_like_term( entropy_intermediate_term::Array{Float64,1},
                            dsf::Array{Float64,1},
                            dsf_default::Array{Float64,1},
                            dfrequency1::Array{Float64,1},
                            dfrequency2::Array{Float64,1}
                            )
    #calculate Kullback-Leibler divergence using trapezoidal rule
    #see https://arxiv.org/pdf/1507.01012.pdf
    #set the intermediate result
    broadcast!((x,y) -> (abs(x)*log(abs(x)/y)),entropy_intermediate_term,dsf,dsf_default)
    #return Kullback-Leibler divergence
    dot(entropy_intermediate_term,dfrequency1) + dot(entropy_intermediate_term,dfrequency2)
end

function quality_of_fit( isf_m::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1},
                         dsf::Array{Float64,1},dsf_default::Array{Float64,1},
                         entropy_intermediate_term::Array{Float64,1},
                         dfrequency1::Array{Float64,1},
                         dfrequency2::Array{Float64,1},
                         regularization_constant::Float64)
    chi_squared_term = mean(broadcast((x,y,z) -> ((x - y)/z)^2,isf,isf_m,isf_error));
    entropy_like_term_fit = regularization_constant*entropy_like_term( entropy_intermediate_term,
                                                                       dsf, dsf_default, dfrequency1,
                                                                       dfrequency2);
    chi_squared_term_fit/2 - entropy_like_term_fit
end

function quality_of_fit!( isf_m::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1},
                          dsf::Array{Float64,1},dsf_default::Array{Float64,1},
                          entropy_intermediate_term::Array{Float64,1},
                          dfrequency1::Array{Float64,1},
                          dfrequency2::Array{Float64,1},
                          regularization_constant::Float64)
    chi_squared_term_fit = mean(broadcast!((x,y,z) -> ((x - y)/z)^2,isf_m,isf,isf_m,isf_error));
    entropy_like_term_fit = regularization_constant*entropy_like_term( entropy_intermediate_term,
                                                                       dsf, dsf_default, dfrequency1,
                                                                       dfrequency2);
    chi_squared_term_fit/2 - entropy_like_term_fit
end

function set_isf_term(imaginary_time::Array{Float64,1},frequency_bins::Array{Float64,1},beta::Float64)
    b = beta;
    isf_term = Array{Float64,2}(undef,size(imaginary_time,1),size(frequency_bins,1));
    for i in 1:size(imaginary_time,1)
        t = imaginary_time[i];
        for j in 1:size(frequency_bins,1)
            f = frequency_bins[j];
            isf_term[i,j] = (exp(-t*f) + exp(-(b - t)*f));
        end
    end
    isf_term
end

function set_isf_trapezoidal!( isf_m::Array{Float64,1}, isf_m2::Array{Float64,1},
                               isf_term::Array{Float64,2}, isf_term2::Array{Float64,2},
                               dsf::Array{Float64,1} 
                               )
    mul!(isf_m,isf_term,abs.(dsf));
    mul!(isf_m2,isf_term2,abs.(dsf));
    isf_m .+= isf_m2;
    isf_m
end

function maxent(dsf::Array{Float64,1},dsf_default::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1},
                frequency::Array{Float64,1},imaginary_time::Array{Float64,1};
                temperature::Float64 = 1.2,
                regularization_constant::Float64 = 0.0,
                number_of_iterations::Int64=1000,
                stop_minimum_fitness::Float64 = 1.0e-8,
                allow_f_increases::Bool=false)
    moment0 = isf[1];
    beta = 1/temperature;

    isf_term = set_isf_term(imaginary_time,frequency,beta);
    isf_term2 = copy(isf_term);

    df = frequency[2:end] .- frequency[1:size(frequency,1) - 1];
    dfrequency1 = zeros(size(frequency,1));
    dfrequency2 = zeros(size(frequency,1));
    for i in 1:(size(frequency,1) - 1)
        dfrequency1[i] = df[i]/2
        dfrequency2[i+1] = df[i]/2
    end
    isf_term .*= dfrequency1';
    isf_term2 .*= dfrequency2';
    
    isf_m = Array{Float64,1}(undef,size(isf,1));
    isf_m2 = Array{Float64}(undef,size(isf,1));

    entropy_intermediate_term = Array{Float64,1}(undef,size(dsf,1));

    result = Optim.optimize(
                 x -> quality_of_fit!(
                     set_isf_trapezoidal!(
                         isf_m, isf_m2, isf_term, isf_term2, x ),
                     isf, isf_error, x, dsf_default, entropy_intermediate_term,
                     dfrequency1, dfrequency2, regularization_constant ),
                 dsf, BFGS(), Optim.Options(
                     iterations=number_of_iterations,
                     allow_f_increases=allow_f_increases));
    
    set_isf_trapezoidal!(isf_m, isf_m2, isf_term, isf_term2, result.minimizer );
    chi_squared_term = mean(broadcast((x,y,z) -> ((x - y)/z)^2,isf,isf_m,isf_error));
    result, chi_squared_term
end
end

