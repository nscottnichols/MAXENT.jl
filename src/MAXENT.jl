module MAXENT
using Random
using Statistics
using LinearAlgebra
using Optim

include("./MAXENT_model_isf.jl")

export maxent

function entropy_like_term(dsf::Array{Float64,1},dsf_default::Array{Float64,1},frequency_bins::Array{Float64,1})
    #see https://arxiv.org/pdf/1507.01012.pdf
    #-MAXENT_model_isf.simps(broadcast((x,y) -> abs(x) - y - (abs(x)*log(abs(x)/y)),dsf,dsf_default),frequency_bins)
    -MAXENT_model_isf.simps(broadcast((x,y) -> (abs(x)*log(abs(x)/y)),dsf,dsf_default),frequency_bins)
end

function quality_of_fit(isf_m::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1},
                        dsf::Array{Float64,1},dsf_default::Array{Float64,1},
                        frequency_bins::Array{Float64,1},regularization_constant::Float64)
    chi_squared_term = mean(broadcast((x,y,z) -> ((x - y)/z)^2,isf,isf_m,isf_error));
    entropy_like_term_fit = regularization_constant*entropy_like_term(dsf,dsf_default,frequency_bins);
    chi_squared_term_fit/2 - entropy_like_term_fit
end

function quality_of_fit!(isf_m::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1},
                         dsf::Array{Float64,1},dsf_default::Array{Float64,1},
                         frequency_bins::Array{Float64,1},regularization_constant::Float64)
    chi_squared_term_fit = mean(broadcast!((x,y,z) -> ((x - y)/z)^2,isf_m,isf,isf_m,isf_error));
    entropy_like_term_fit = regularization_constant*entropy_like_term(dsf,dsf_default,frequency_bins);
    chi_squared_term_fit/2 - entropy_like_term_fit
end

function maxent(dsf::Array{Float64,1},dsf_default::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1},
                frequency_bins::Array{Float64,1},imaginary_time::Array{Float64,1};
                temperature::Float64 = 1.2,
                regularization_constant::Float64 = 0.0,
                isf_m_type::String="simps",
                number_of_iterations::Int64=1000,
                stop_minimum_fitness::Float64 = 1.0e-8,
                allow_f_increases::Bool=true)
    moment0 = isf[1];
    beta = 1/temperature;
    isf_m_function = getfield(MAXENT_model_isf, Symbol(isf_m_type));
    isf_m = Array{Float64,1}(undef,size(isf,1));

    result = Optim.optimize(x -> quality_of_fit!(isf_m_function(isf_m,x,frequency_bins,imaginary_time,beta),isf,isf_error,x,dsf_default,frequency_bins,regularization_constant),dsf,BFGS(), Optim.Options(iterations=number_of_iterations,allow_f_increases=allow_f_increases));
    result
end
end

