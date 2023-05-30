addpath('../helper_functions')
addpath('../helper_functions/event_detection')
% Using the generated result .mat files after running main.m, this script
% will generate the average results in an excel file

w = 2; %event size selector
window = 1; %window size selector
conf = 2; %confidence parameter (C)
Ms = [50, 20];
M = Ms(window);

savedFileName = strcat('Experiment_Results/allmethods_results_w', int2str(M), '_conf_',int2str(conf),"_", int2str(w), '.mat');
load(savedFileName);
nots = 2000;
event_loc = nots/2;

dys = [7.5e-4, 7.5e-6;
       1.25e-3, 1.25e-5];

dy = dys(window, w);
number_of_experiments = 100;

%[sum_threshold, sum_tensorflow, sum_ensemble, sum_MATLABnn, sum_XGB, sum_NN, sum_CNN, sum_XGB2] = sumInitalizers();
kalman_res = [store_threshold; store_tensorflow; store_ensemble; store_matlabnn; store_xgboost];
blackbox_res = [store_blackboxnn; store_blackboxcnn; store_blackboxxgb];

kalman_methods = size(kalman_res,1);
bb_methods = size(blackbox_res,1);

clear savedFileName store_blackboxxgb store_blackboxcnn store_blackboxnn
clear store_ensemble store_matlabnn store_tensorflow store_threshold store_xgboost

%% extract relevant data out of structs
for i = 1:kalman_methods % for each kalman method
    for j = 1:size(kalman_res(1,:),2) % for each experiment
        store_ext(j) = extract_kalman_filter_results(kalman_res(i, j), nots, M); % of method i, exp j
    end
    kalman_res_ext(i,:) = store_ext; % exctracted relevant data out of kalman_res
end
%% average the parameters for all runs
clearvars -except kalman_res_ext blackbox_res kalman_methods bb_methods number_of_experiments M event_loc dy w
total_methods = kalman_methods + bb_methods;

% initialize table columns
run_time = [];
False_detection = []; % 8 for each method (5+3)
Miss = [];
Detection_delay = [];
mean_k0 = [];
rmse_loc = [];
rmse_size = [];
rmse_size_delay = []; % NA for blackbox
k0 = [];
k = [];

for i=1:number_of_experiments % each experiment is its own column
    
    run_time = [run_time, [kalman_res_ext(1,i).time_elapsed; kalman_res_ext(2,i).time_elapsed; kalman_res_ext(3,i).time_elapsed; kalman_res_ext(4,i).time_elapsed; kalman_res_ext(5,i).time_elapsed; blackbox_res(1,i).time_elapsed; blackbox_res(2,i).time_elapsed; blackbox_res(3,i).time_elapsed]];
    False_detection = [False_detection, [kalman_res_ext(1,i).false_detection; kalman_res_ext(2,i).false_detection; kalman_res_ext(3,i).false_detection; kalman_res_ext(4,i).false_detection; kalman_res_ext(5,i).false_detection; blackbox_res(1,i).false_detection; blackbox_res(2,i).false_detection; blackbox_res(3,i).false_detection]];
    Miss = [Miss, [kalman_res_ext(1,i).no_detection; kalman_res_ext(2,i).no_detection; kalman_res_ext(3,i).no_detection; kalman_res_ext(4,i).no_detection; kalman_res_ext(5,i).no_detection; blackbox_res(1,i).no_detection; blackbox_res(2,i).no_detection; blackbox_res(3,i).no_detection]];

    % find which samples have miss or false_detection to find error
    erroneous_detection = find((Miss(:,end) + False_detection(:,end)) > 0);
   
    k = [k, [kalman_res_ext(1,i).detected_event_time_k; kalman_res_ext(2,i).detected_event_time_k; kalman_res_ext(3,i).detected_event_time_k; kalman_res_ext(4,i).detected_event_time_k; kalman_res_ext(5,i).detected_event_time_k; blackbox_res(1,i).window_start+M; blackbox_res(2,i).window_start+M; blackbox_res(3,i).window_start+M]];
    k0 = [k0, [kalman_res_ext(1,i).detected_event_time_k0; kalman_res_ext(2,i).detected_event_time_k0; kalman_res_ext(3,i).detected_event_time_k0; kalman_res_ext(4,i).detected_event_time_k0; kalman_res_ext(5,i).detected_event_time_k0; blackbox_res(1,i).detected_time; blackbox_res(2,i).detected_time; blackbox_res(3,i).detected_time]];
    k0(erroneous_detection, end) = -1;

    Detection_delay = [Detection_delay, (abs(k(:,end)-k0(:,end)))];
    Detection_delay(erroneous_detection, end) = -1;

    rmse_loc = [rmse_loc, ((k0(:,end)-event_loc).^2)]; % take sqrt later
    rmse_loc(erroneous_detection, end) = -1;

    dy_hat = [kalman_res_ext(1,i).event_size; kalman_res_ext(2,i).event_size; kalman_res_ext(3,i).event_size; kalman_res_ext(4,i).event_size; kalman_res_ext(5,i).event_size; blackbox_res(1,i).event_size; blackbox_res(2,i).event_size; blackbox_res(3,i).event_size];
    rmse_size = [rmse_size, ((dy_hat-dy).^2)]; % take sqrt later
    rmse_size(erroneous_detection, end) = -1;

    dy_hat_delay = [kalman_res_ext(1,i).event_size_after_delay; kalman_res_ext(2,i).event_size_after_delay; kalman_res_ext(3,i).event_size_after_delay; kalman_res_ext(4,i).event_size_after_delay; kalman_res_ext(5,i).event_size_after_delay];
    rmse_size_delay = [rmse_size_delay, ((dy_hat_delay-dy).^2)]; % take sqrt later
    rmse_size_delay(erroneous_detection(erroneous_detection<=kalman_methods), end) = -1;
end

% post processing
run_time = mean(run_time, 2);
%run_time = run_time ./run_time(4); % Normalize wrt MatlabNN

False_detection = sum(False_detection, 2);
False_detection = (100/number_of_experiments) * False_detection; % convert to percent

Miss = sum(Miss, 2);
Miss = (100/number_of_experiments) * Miss; % convert to percent

% selective mean to consider non -1 entries only
k0 = selective_mean(k0);
Detection_delay = selective_mean(Detection_delay);
rmse_loc = sqrt(selective_mean(rmse_loc));
rmse_size = sqrt(selective_mean(rmse_size))/dy;
rmse_size_delay = sqrt(selective_mean(rmse_size_delay))/dy;
rmse_size_delay = [rmse_size_delay; "NA"; "NA"; "NA"];
%% Table creator
methods = ["threshold","tensorflow","ensemble","MATLABnn","XGBoost"];
methods2 = ["NN","CNN","XGBoost"];

Methods = [methods(1); methods(2); methods(3); methods(4); methods(5); methods2(1); methods2(2); methods2(3)];
results = table(Methods, run_time, False_detection, Miss, Detection_delay, k0, rmse_loc, rmse_size, rmse_size_delay);

fprintf("resulting table with %d experiments for event sizes %f: \n",number_of_experiments,dy);
display(results)
filename = strcat('Experiment_Results/results_w',string(M),".xlsx"); % excel file for lists
writetable(results, filename, 'Sheet', w)


