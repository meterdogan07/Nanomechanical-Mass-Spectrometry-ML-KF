function [event] = bruteforce_regression(rnet_name, r_net, data, M)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    if(rnet_name == "XGBoost")
        event = single(pyrunfile("./Blackbox_ML/regression_bruteforce_xgboost.py", "event", x=data, M=M));
    else
        event = predict(r_net, data);
    end
end

