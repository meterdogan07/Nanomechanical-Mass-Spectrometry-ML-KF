
% Optimal Threshold selection algorithm
addpath('../helper_functions')
addpath('../helper_functions/event_detection')
window_size = 20;
nots = 2000;
M = window_size;

if M==50 %if M=50 
    event_size = 7.5e-6;
elseif M==20 %if M=20 
    event_size = 1.25e-5;
end
%---------------------------------------------------
%Kalman Experiments
threshold = 125; % LargeVal
past_th = threshold;
low_th = 0;
number_of_experiments = 3;
results1 = [];
keep1 = 1;

while(keep1 == 1)
    fprintf("threshold = %f, low_th = %f \n",threshold,low_th);
    if(threshold-low_th<1.5)
        keep1 = 0;
        break
    end
    
    [run_time, False_detection, Miss, Detection_delay, k0, rmse_loc, rmse_size, rmse_size_delay] = Simulation(nots,window_size,event_size,threshold,number_of_experiments);
    results1 = [results1; window_size, threshold, event_size, run_time, False_detection, Miss, Detection_delay, k0, rmse_loc, rmse_size, rmse_size_delay];
    fprintf("run time: %f \nfalse detection: %d \nmiss: %d \ndetection delay: %f \nk0: %f \n",run_time, False_detection, Miss, Detection_delay, k0);
    fprintf("rmse loc: %f \nrmse size: %f \nrmse size (delayed): %f \n", rmse_loc, rmse_size, rmse_size_delay);

    if(size(results1,1) ~= 0)
        if(isBetter(results1))
            past_th = threshold;
            threshold = threshold-floor((threshold-low_th)/2);
        else
            low_th = threshold;
            threshold = past_th-floor((past_th-low_th)/2);
            results1 = results1(1:end-1,:);
        end
    else
        past_th = threshold;
        threshold = round(threshold/2);
    end
end

%%
% Trial on one experiment
nots = 2000;
M = window_size;
results = [];

if M==50 %if M=50 
    event_size = 7.5e-6;
elseif M==20 %if M=20 
    event_size = 1.25e-5;
else
    event_size = 2e-5;
end

%------------------------------------------------------------------------------------------------
%Kalman Experiments
threshold = past_th;
number_of_experiments = 100;
[run_time, False_detection, Miss, Detection_delay, k0, rmse_loc, rmse_size, rmse_size_delay] = Simulation(nots,window_size,event_size,threshold,number_of_experiments);
fprintf("run time: %f \nfalse detection: %d \nmiss: %d \ndetection delay: %f \nk0: %f \n",run_time, False_detection, Miss, Detection_delay, k0);
fprintf("rmse loc: %f \nrmse size: %f \nrmse size (delayed): %f \n", rmse_loc, rmse_size, rmse_size_delay);


%%
%------------------------------------------------------------------------------------------------

% Simulation for Kalman Filter
function [run_time, False_detection, Miss, Detection_delay, k0, rmse_loc, rmse_size, rmse_size_delay] = Simulation(nots,window_size,eventSize,threshold,number_of_experiments)
%Kalman Experiments
    M = window_size;
    eventPlace = nots/2;
    dy = eventSize; %Event size (fractional freq shift caused by the event)
    run('initialize_kalman_variables.m');
    %sensor simulation
    for i=1:number_of_experiments
        %fprintf("experiment: %d, with event size: %.7f \n",i,dy)
        [y_ro, y_state] = generata_sensor_data(nots,eventPlace,dy,params);
        store_threshold(i) = run_kalman_simulation("threshold", y_ro, threshold, window_size, 0.5);
        kalman_res_ext(i) = extract_kalman_filter_results(store_threshold(i), nots, M);
    end
    [run_time, False_detection, Miss, Detection_delay, k0, rmse_loc, rmse_size, rmse_size_delay] = extract_average_results(kalman_res_ext, number_of_experiments, eventPlace, dy);
end


%Determine whether a threshold is better than another
function [better] = isBetter(results1)
    if(results1(end,6)==100)
        better = 1;
    elseif(results1(end,6)==results1(end-1,6) && results1(end,5)==results1(end-1,5))
        if(results1(end,9)<results1(end-1,9))
            better = 1;
        else
            better = 0;
        end
    elseif(results1(end,6)<=results1(end-1,6) && results1(end,5)<=results1(end-1,5))
        better = 1;
    else
        better = 0;
    end
end
