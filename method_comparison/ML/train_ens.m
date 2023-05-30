function [ens] = train_ens(data, label, M)
    ens = fitcensemble(data,label, 'Method', 'RUSBoost',...
        'NumLearningCycles', M, 'Learners', 'Tree','NPrint',5);
end
