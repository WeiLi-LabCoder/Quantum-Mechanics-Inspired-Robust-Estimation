function X = make_signal(cfg, N_t, L)
    switch lower(cfg.sigModel)
        case 'gmm'
            X = zeros(N_t, L);
            mask = rand(N_t, L) < cfg.pi_mix(2);
            X(mask)  = cfg.sigma_mix(2) * randn(nnz(mask),  1);
            X(~mask) = cfg.sigma_mix(1) * randn(nnz(~mask), 1);
        case 'gaussian'
            X = randn(N_t, L);
        case 'noprior'
            X = 3 + 10*randn(N_t, L);
        case 'qam'
            M = get_mod_order(cfg.mod);
            idx = randi([0 M-1], N_t, L);
            X = qammod(idx, M, 'UnitAveragePower', true);
            if ~(isfield(cfg,'isComplex') && cfg.isComplex)
                X = real(X);
            end
        otherwise
            error('Unknown cfg.sigModel');
    end
end