function H = make_channel(cfg, N_r, N_t, var_channel)
    if isfield(cfg,'isComplex') && cfg.isComplex
        H_w = (randn(N_r,N_t) + 1j*randn(N_r,N_t))/sqrt(2);
    else
        H_w = randn(N_r,N_t);
    end

    if isfield(cfg,'useCorrChan') && cfg.useCorrChan
        Rr = toeplitz(cfg.rho_r.^(0:N_r-1));
        Rt = toeplitz(cfg.rho_t.^(0:N_t-1));
        Rr = (Rr+Rr')/2; Rt=(Rt+Rt')/2;
        Rr12 = sqrtm(Rr); Rt12 = sqrtm(Rt);
        H_w = Rr12 * H_w * Rt12;
    end

    if isfield(cfg,'fadingModel') && strcmpi(cfg.fadingModel,'rician')
        K = cfg.K_rician;
        a_r = ones(N_r,1); a_t = ones(N_t,1);
        H_los = a_r*a_t.';
        if ~isreal(H_w), H_los = complex(H_los,0); end
        H = sqrt(K/(K+1))*H_los + sqrt(1/(K+1))*H_w;
    else
        H = H_w;
    end

    H = sqrt(var_channel) * H;
end