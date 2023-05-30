%stacker----stacks 3D data into 2D data by adding 
function [n_data] = stacker(data)
    n_data = zeros(length(data(:,1,1))*length(data(1,:,1)),length(data(1,1,:)));
    for i=1:length(data(:,1,1))
            n_data(((i-1)*length(data(1,:,1))+1):(i*length(data(1,:,1))),:) = data(i,:,:);
    end
end