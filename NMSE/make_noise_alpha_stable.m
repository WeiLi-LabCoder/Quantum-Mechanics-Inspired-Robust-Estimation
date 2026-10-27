function N = make_noise_alpha_stable(cfg, N_r, L, noise_power, delta)
    gamma = (noise_power/2)^(1/cfg.alpha);
    N = sAS(cfg.alpha, gamma, N_r, L, delta);

    if isfield(cfg,'isComplex') && cfg.isComplex
        N_im = sAS(cfg.alpha, gamma, N_r, L, delta);
        N = (N + 1j*N_im)/sqrt(2);
    end
 
end


function X = sAS(alpha, gamma, m, n, delta)
    if alpha<=0 || alpha>2, error('alpha in (0,2]'); end
    if nargin<5, delta=0; end
    if abs(alpha-2)<1e-12
        X = delta + sqrt(2)*gamma*randn(m,n);
        return;
    end
    U = pi*(rand(m,n)-0.5);
    W = exprnd(1, m, n);
    X0 = gamma .* (sin(alpha.*U) ./ (cos(U).^(1/alpha))) .* ...
         ((cos((1-alpha).*U) ./ W).^((1-alpha)/alpha));
    X = X0 + delta;
end