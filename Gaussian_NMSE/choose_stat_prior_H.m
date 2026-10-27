function [pri_mean, pri_var] = choose_stat_prior_H(cfg, stats)
%CHOOSE_STAT_PRIOR_H
% Generate prior mean/variance for channel estimation experiments
%
% Inputs
%   cfg.priorCase : 'known' | 'partial' | 'absent'
%   stats         : output struct from make_channel()
%   var_channel   : true channel power
%
% Outputs
%   pri_mean
%   pri_var

% -------- true statistics (empirical) --------
true_mean = stats.emp_mean;
true_var  = stats.emp_var;

true_var = max(real(true_var),1e-12);

switch lower(cfg.priorCase)

    %% ==================================================
    %% 1. Correct prior (matched)
    %% ==================================================
    case 'known'

        pri_mean = true_mean;
        pri_var  = true_var;

    %% ==================================================
    %% 2. Mismatched prior
    %% ==================================================
    case 'partial'

        if ~isfield(cfg,'partialMode')
            cfg.partialMode = 'var_mismatch';
        end

        pri_mean = true_mean;
        pri_var  = true_var;

        switch lower(cfg.partialMode)

            % -------- variance mismatch --------
            case 'var_mismatch'

                if ~isfield(cfg,'varMismatchFactor')
                    cfg.varMismatchFactor = 2;
                end

                pri_var = cfg.varMismatchFactor * true_var;

            % -------- mean mismatch --------
            case 'mean_mismatch'

                if ~isfield(cfg,'meanMismatchBias')
                    cfg.meanMismatchBias = 0.5;
                end

                pri_mean = true_mean + cfg.meanMismatchBias;

            % -------- both mismatch --------
            case 'both'

                if ~isfield(cfg,'varMismatchFactor')
                    cfg.varMismatchFactor = 2;
                end

                if ~isfield(cfg,'meanMismatchBias')
                    cfg.meanMismatchBias = 0.5;
                end

                pri_var  = cfg.varMismatchFactor * true_var;
                pri_mean = true_mean + cfg.meanMismatchBias;

            otherwise
                error('Unknown partialMode');

        end


    %% ==================================================
    %% 3. Weak / no prior
    %% ==================================================
    case 'absent'

        if ~isfield(cfg,'absentMode')
            cfg.absentMode = 'weak';
        end

        switch lower(cfg.absentMode)

            % default Gaussian prior
            case 'default01'

                pri_mean = 0;
                pri_var  = 1;

            % weak prior (≈ no information)
            case 'weak'

                if ~isfield(cfg,'weakPriorVar')
                    cfg.weakPriorVar = 1e6;
                end

                pri_mean = 0;
                pri_var  = cfg.weakPriorVar;

            otherwise
                error('Unknown absentMode');

        end

    otherwise
        error('Unknown priorCase')

end

end