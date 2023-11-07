clearvars -except structdata structlabel struct_tensorflow struct_ensemble struct_MATLAB_nn data label training_dataset blackbox_dataset struct_blackbox % keep the trained models to reduce runtime
addpath('ML')
addpath('Blackbox_ML')
addpath('Training_Data_ML_aided_KF')
addpath('../helper_functions')
addpath('../helper_functions/event_detection')

window_size = 50;
M = window_size;
run('initialize_kalman_variables.m');
params; %params struct variable is defined in "initialize_vars.m"

nots = 2000; % number of time steps
eventPlace = nots/2;

fieldname = matlab.lang.makeValidName(strcat('wlength_',int2str(window_size)));


%generate labeled likelihood data using event detection algorithm and kalman filter
training_dataset.a = 1;
if(~isfield(training_dataset,fieldname) && training_dataset.a == 1)
    [data, label] = generate_train_data_kalman(window_size, params); 
    structdata.(fieldname) = data;
    structlabel.(fieldname) = label;    
end

%%

% train XGBoost model
pyrunfile("./ML/train_xgboost.py", X=structdata.(fieldname),y=structlabel.(fieldname), M=window_size)


% train Tensorflow model
struct_tensorflow.a = 1;
if (~isfield(struct_tensorflow,fieldname) || struct_tensorflow.a == 1)
    tic
    pyrunfile("./ML/tf_kalman_model.py", X=structdata.(fieldname), y=structlabel.(fieldname), M=window_size)
    tensorflow_train_ime = toc;
end
struct_tensorflow.(fieldname) = importKerasNetwork(strcat('./ML/saved_models/kalman_tf_',int2str(window_size),'.h5'));


% train Ensemble RUSBoost
struct_ensemble.a = 1;
if (~isfield(struct_ensemble,fieldname) || struct_ensemble.a == 1)
    tic
    struct_ensemble.(fieldname) = train_ens(data, label, window_size);
    ens_train_ime = toc;
end

% train MATLAB NN
struct_MATLAB_nn.a = 1;
if (~isfield(struct_MATLAB_nn,fieldname) || struct_MATLAB_nn.a == 1)
    tic
    struct_MATLAB_nn.(fieldname) = train_MATLAB_nn(data, label, window_size);
    Mnn_train_time = toc;
end

%%
% BLACKBOX ML
% create dataset with raw data for blackbox
blackbox_dataset.a = 1;

if(blackbox_dataset.a == 1)
    blackbox_dataset.all.(fieldname) = data_loader(window_size,1, params);
    blackbox_dataset.little.(fieldname) = data_loader(window_size,0.1, params);
end

struct_blackbox.a = 1;
if (~isfield(struct_blackbox,fieldname) || struct_blackbox.a == 1)
    pyrunfile("./Blackbox_ML/NN.py", dataset = blackbox_dataset.little.(fieldname), dataset2 = blackbox_dataset.all.(fieldname), M=window_size)
    pyrunfile("./Blackbox_ML/CNN.py", dataset = blackbox_dataset.little.(fieldname), dataset2 = blackbox_dataset.all.(fieldname), M=window_size)
    pyrunfile("./Blackbox_ML/XGBoost.py", dataset = blackbox_dataset.little.(fieldname), dataset2 = blackbox_dataset.all.(fieldname), M=window_size)
end

struct_blackbox.(fieldname).NN_classifier = importKerasNetwork(strcat("./Blackbox_ML/classification_nn_",int2str(window_size),".h5"));
struct_blackbox.(fieldname).NN_regressor = importKerasNetwork(strcat("./Blackbox_ML/regression_nn_",int2str(window_size),".h5"));
struct_blackbox.(fieldname).CNN_classifier = importKerasNetwork(strcat("./Blackbox_ML/classification_cnn_",int2str(window_size),".h5"));
struct_blackbox.(fieldname).CNN_regressor = importKerasNetwork(strcat("./Blackbox_ML/regression_cnn_",int2str(window_size),".h5"));


%%
%Experiments with all methods:
addpath('ML')
addpath('../helper_functions')
addpath('../helper_functions/event_detection')

if M==50 %if M=50 
    event_sizes = [7.5e-4, 7.5e-6];
    threshold_val = 23;
else %if M=20 
    event_sizes = [1.25e-3, 1.25e-5];
    threshold_val = 19;
end

confidences = [1/2]; % try different conficence (C) values 
number_of_experiments = 100; %1000

for conf = 1:size(confidences,2)
    fname = strcat('Experiment_Results/allmethods_results_w',string(window_size),"_conf_",string(conf));
    %------------------------------------------------------------------------------------------------
    % ML Aided Kalman Experiments
    for eventSize = 1:size(event_sizes,2)
        dy = event_sizes(eventSize); %Event size (fractional freq shift caused by the event)
        for i=1:number_of_experiments
            %sensor simulation
            fprintf("experiment: %d, with event size: %f \n",i,dy)
            [y_ro, y_state] = generata_sensor_data(nots,eventPlace,dy,params);
    
            % perform test on the data
            methods = ["threshold","tensorflow","ensemble","MATLABnn","XGBoost"];
            store_threshold(i) = run_kalman_simulation(methods(1), y_ro, threshold_val, window_size, confidences(conf));
            store_tensorflow(i) = run_kalman_simulation(methods(2), y_ro, struct_tensorflow.(fieldname), window_size, confidences(conf));
            store_ensemble(i) = run_kalman_simulation(methods(3), y_ro, struct_ensemble.(fieldname), window_size, confidences(conf));
            store_matlabnn(i) = run_kalman_simulation(methods(4), y_ro, struct_MATLAB_nn.(fieldname), window_size, confidences(conf));
            store_xgboost(i) = run_kalman_simulation(methods(5), y_ro, 1, window_size, confidences(conf));

            methods2 = ["NN","CNN","XGBoost"];
            store_blackboxnn(i) = run_blackbox_simulation(methods2(1), methods2(1), struct_blackbox.(fieldname).NN_classifier, struct_blackbox.(fieldname).NN_regressor, y_ro, window_size,eventPlace);
            store_blackboxcnn(i) = run_blackbox_simulation(methods2(2), methods2(2), struct_blackbox.(fieldname).CNN_classifier, struct_blackbox.(fieldname).CNN_regressor, y_ro, window_size,eventPlace);
            store_blackboxxgb(i) = run_blackbox_simulation(methods2(3), methods2(3), 1, 1, y_ro, window_size,eventPlace);
        end
        filename = strcat(fname,"_",string(eventSize),".mat"); % save all variables
        save(filename,'store_threshold','store_tensorflow','store_ensemble','store_matlabnn','store_xgboost','store_blackboxnn','store_blackboxcnn','store_blackboxxgb')
    end
end
