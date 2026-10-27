%% ================= Helpers =================
function [pri_mean, pri_var] = choose_stat_prior_X(cfg, Symbol_X)
    true_mean = mean(Symbol_X(:));
    true_var  = var(Symbol_X(:), 1);
    true_var = max(true_var, 1e-12);

    switch lower(cfg.priorCase)
        case 'known'
            pri_mean = true_mean;
            pri_var  = true_var;

        case 'partial'
            pm = true_mean; pv = true_var;
            if ~isfield(cfg,'partialMode'), cfg.partialMode='var_mismatch'; end
            switch lower(cfg.partialMode)
                case 'var_mismatch'
                    f = cfg.varMismatchFactor;
                    pri_mean = pm;
                    pri_var  = max(f * pv, 1e-12);
                case 'mean_mismatch'
                    b = cfg.meanMismatchBias;
                    pri_mean = pm + b;
                    pri_var  = pv;
                case 'both'
                    f = cfg.varMismatchFactor; b = cfg.meanMismatchBias;
                    pri_mean = pm + b;
                    pri_var  = max(f * pv, 1e-12);
                otherwise
                    error('Unknown cfg.partialMode');
            end

        case 'absent'
            if ~isfield(cfg,'absentMode'), cfg.absentMode='default01'; end
            switch lower(cfg.absentMode)
                case 'default01'
                    pri_mean = 0; pri_var = 1;
                case 'weak'
                    pri_mean = 0; pri_var = cfg.weakPriorVar;
                otherwise
                    error('Unknown cfg.absentMode');
            end

        otherwise
            error('Unknown cfg.priorCase');
    end
end