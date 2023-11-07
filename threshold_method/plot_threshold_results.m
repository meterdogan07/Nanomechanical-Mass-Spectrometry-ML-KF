%Check experiments around diferent thresholds
addpath('../helper_functions')
addpath('../helper_functions/event_detection')

window_size = 50;
nots = 2000;
M = window_size;
event_sizes = [7.5e-4, 5e-4, 2.5e-4, 1e-4, 7.5e-5, 5e-5, 2.5e-5, 1e-5, 7.5e-6, 5e-6, 2.5e-6, 1e-6];

%------------------------------------------------------------------------------------------------
%Kalman Experiments
number_of_experiments = 20;
results1 = [];
% results2 = [];

                                                               
for event_size = event_sizes
    for threshold = 5:5:120
        [run_time, False_detection, Miss, Detection_delay, k0, rmse_loc, rmse_size, rmse_size_delay] = Simulation(nots,window_size,event_size,threshold,number_of_experiments);
        results1 = [results1; event_size, window_size, threshold, event_sizes(1), run_time, False_detection, Miss, Detection_delay, k0, rmse_loc, rmse_size, rmse_size_delay];
        fprintf("run time: %f \nfalse detection: %d \nmiss: %d \ndetection delay: %f \nk0: %f \n",run_time, False_detection, Miss, Detection_delay, k0);
        fprintf("rmse loc: %f \nrmse size: %f \nrmse size (delayed): %f \n", rmse_loc, rmse_size, rmse_size_delay);
    end
end
save("threshold_results",'results1')
% ------------------------------------------------------------------------------
%%
%Plot results
es = [7.5e-4, 5e-4, 2.5e-4, 1e-4, 7.5e-5, 5e-5, 2.5e-5, 1e-5, 7.5e-6, 5e-6, 2.5e-6, 1e-6];

a = zeros(24,12);
for i = 0:11
    a(:,i+1) = results1(i*24+1:i*24+24,6);
end
false_detection = a; % False detections

a = zeros(24,12);
for i = 0:11
    a(:,i+1) = results1(i*24+1:i*24+24,7);
end
miss= a; % Miss


figure
set(gca,"FontSize",50)
thresholds = 10:10:110;
contourf(miss + false_detection, 6, "LineWidth", 0.01)
c = colorbar;  
c.Ruler.TickLabelFormat='%g%%';

yticks(2:2:22)
yticklabels(thresholds)

xticks(1:12)
xticklabels(es)
title("False Detection + Miss",'FontSize', 35, 'FontName', "Times")
grid on
xlabel("Relative Event Size",'FontSize', 35, 'FontName', "Times")
ylabel("Threshold Value",'FontSize', 35, 'FontName', "Times")


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

