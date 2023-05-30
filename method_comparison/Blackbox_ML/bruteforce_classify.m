function [event] = bruteforce_classify(cnet_name, c_net, data, M)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    if(cnet_name == "XGBoost")
        event = pyrunfile("./Blackbox_ML/classifier_bruteforce_xgboost.py", "event", x=data, M=M);
    else
        event = predict(c_net, data);
    end
end

