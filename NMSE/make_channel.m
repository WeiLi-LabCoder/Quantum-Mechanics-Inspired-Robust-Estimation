function [H, stats] = make_channel(cfg, N_r, N_t, var_channel)
%MAKE_CHANNEL  Physically reasonable complex MIMO channel  
% Outputs:
%   H     : N_r x N_t complex channel
%   stats : struct with per-element mean/var/power (empirical, for debugging)


    %  ----------------------------
    % 1) Base i.i.d. complex Gaussian (Rayleigh scatter)
    % ----------------------------
    Hw = (randn(N_r,N_t) + 1j*randn(N_r,N_t))/sqrt(2); % CN(0,1)

    % Optional: make scatter variance noticeably different (for mismatch experiments)
    % This changes the *raw* scatter distribution; later we normalize total power.
    Hw = sqrt(max(cfg.trueVarScale, 1e-12)) * Hw;

    % ----------------------------
    % 2) Spatial correlation (Kronecker), applied to scatter component
    % ----------------------------
    if cfg.useCorrChan
        Rr = toeplitz(cfg.rho_r.^(0:N_r-1));
        Rt = toeplitz(cfg.rho_t.^(0:N_t-1));

        % symmetrize for numerical stability
        Rr = (Rr + Rr')/2;
        Rt = (Rt + Rt')/2;

        % Use Cholesky when possible (faster + stable). Fall back to sqrtm.
        [Rr12, pr1] = chol(Rr, 'lower');
        if pr1 ~= 0, Rr12 = sqrtm(Rr); end

        [Rt12, pr2] = chol(Rt, 'lower');
        if pr2 ~= 0, Rt12 = sqrtm(Rt); end

        % Note: Rt12' here ensures correct covariance Rt when Hw is i.i.d.
        Hw = Rr12 * Hw * Rt12';
    end

    % ----------------------------
    % 3) Fading model: Rayleigh or Rician
    % ----------------------------
    switch lower(cfg.fadingModel)
        case 'rayleigh'
            H = Hw;

        case 'rician'
            K = max(cfg.K_rician, 0);

            % Build a more physical LOS component: rank-1 steering (ULA)
            if cfg.useULAsteering
                % random AoA/AoD if not provided
                if isempty(cfg.AoA), theta_r = 2*pi*rand; else, theta_r = cfg.AoA; end
                if isempty(cfg.AoD), theta_t = 2*pi*rand; else, theta_t = cfg.AoD; end

                nr = (0:N_r-1).';
                nt = (0:N_t-1).';

                d = cfg.d_over_lambda;
                a_r = exp(1j*2*pi*d*nr*sin(theta_r));  % N_r x 1
                a_t = exp(1j*2*pi*d*nt*sin(theta_t));  % N_t x 1

                H_los = a_r * a_t.';  % N_r x N_t (rank-1)
                % Normalize LOS so its average element power is 1
                H_los = H_los / sqrt(mean(abs(H_los(:)).^2));
            else
                % fallback LOS: all-ones (still rank-1)
                H_los = ones(N_r,N_t);
            end

            % Rician mix (LOS + scatter)
            H = sqrt(K/(K+1))*H_los + sqrt(1/(K+1))*Hw;

        otherwise
            error('make_channel: unknown fadingModel "%s". Use rayleigh|rician.', cfg.fadingModel);
    end

    % Optional: deterministic complex bias to force nonzero mean even in Rayleigh
    if cfg.forceMeanBias ~= 0
        H = H + cfg.forceMeanBias;
    end

    % ----------------------------
    % 4) Power normalization: E|H_ij|^2 = var_channel
    % ----------------------------
    p = mean(abs(H(:)).^2);
    if ~isfinite(p) || p <= 0
        error('make_channel: invalid power encountered (p=%g).', p);
    end
    H = H / sqrt(p) * sqrt(var_channel);

    % ----------------------------
    % 5) Return empirical stats (useful for verifying mismatch)
    % ----------------------------
    stats = struct();
    stats.emp_mean  = mean(H(:));
    stats.emp_var   = var(H(:), 1);              % complex variance definition via MATLAB var
    stats.emp_power = mean(abs(H(:)).^2);        % should be var_channel
end