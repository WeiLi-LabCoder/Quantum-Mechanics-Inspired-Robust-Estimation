function N = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta)
    gamma = (var_noise/2)^(1/cfg.alpha);
    N = sAS(cfg.alpha, gamma, N_r, L, delta);

    if isfield(cfg,'isComplex') && cfg.isComplex
        N_im = sAS(cfg.alpha, gamma, N_r, L, delta);
        N = (N + 1j*N_im)/sqrt(2);
    end

    if cfg.useColored
        if isfield(cfg,'colorMode') && strcmpi(cfg.colorMode,'spatial')
            if isfield(cfg,'Rn') && ~isempty(cfg.Rn)
                Rn = cfg.Rn;
            else
                if ~isfield(cfg,'rho'), cfg.rho = 0.7; end
                idx = (1:N_r)';
                Rn  = cfg.rho .^ abs(idx - idx');
            end
            Rn = (Rn + Rn')/2;
            [V,D] = eig(Rn);
            d = max(real(diag(D)),0);
            C = V * diag(sqrt(d)) * V';
            N = C * N;
            N = N * (sqrt(var_noise) / max(std(N(:)), eps));
        else
            for r = 1:N_r
                N(r,:) = filter(cfg.h, 1, N(r,:));
            end
            N = N * (sqrt(var_noise) / max(std(N(:)), eps));
        end
    end
end