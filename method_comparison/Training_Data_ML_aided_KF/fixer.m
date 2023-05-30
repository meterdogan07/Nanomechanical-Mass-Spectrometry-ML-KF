% deletes labels for current trial after an event has occurred
function [frame, label] = fixer(l, lbl, nots_min_M)
    
    event_occurred = 0;
    
    good_ixs = [];
    for k = 1:length(lbl)
        if(lbl(k) == 0 && ~event_occurred)
            good_ixs = [good_ixs k];
        end
        if(lbl(k) == 1)
            good_ixs = [good_ixs k];
            event_occurred = 1;
        end
        if(mod(k, nots_min_M) == 0)
            event_occurred = 0;
        end
    end
    
    frame = l(good_ixs, :);
    label = lbl(good_ixs,:);
end