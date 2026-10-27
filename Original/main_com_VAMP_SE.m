clear; clc;  rng(1); 

% ---------- Dimensions ----------
N_r = 1000; N_t = 1000; L = 1;
SNR_dB = 0; 

% ---------- Channel power ----------
var_channel = 1;

% ---------- Config (one place to change) ----------
cfg.alpha      = 2;        % 0<alpha<=2; alpha=2 -> Gaussian
cfg.useColored = false;      % color the noise?
cfg.h          = [1 -1];     % FIR for colored noise (e.g., high-pass)

cfg.sigModel   = 'gmm';      % 'gmm' | 'gaussian' | 'noprior'
cfg.pi_mix     = [0.5 0.5];  % GMM mixing weights (sum=1)
cfg.sigma_mix  = [1 10];     % GMM stds, zero-mean components

% ---------- For α-stable noise ----------
P_sig = 1;     % target signal power for SNR definition
delta = 0;     % location (mean) for noise

var_noise = P_sig / 10^(SNR_dB/10);
pri_var = 1;
pri_mean = 0;

noise = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta);

% ===== Channel =====
measurement = sqrt(var_channel) * randn(N_r, N_t);

% ===== Tx Signal (all standardized to mean=0, var≈1) =====
Symbol_X = make_signal(cfg, N_t, L);

% ===== I/O =====
Chan_Symbol = measurement * Symbol_X;
obser       = Chan_Symbol + noise;

[length_row,length_col] = size(measurement);

%% ================== SVD & y_flag (Alg. step 2) ==================
[U,S,V] = svd(measurement);
y_flag  = (1/var_noise) * (S' * (U' * obser));   % \bar{y} = sigma0^{-2} S^H U^H y~

%% ================== First LMMSE init (steps 1&2) ==================
gama_2(1,1)  = 1/pri_var;
x2_mean(:,1) = var_noise * ...
    (measurement' * measurement * var_noise + gama_2(1,1)*eye(length_col)) \ ...
    (measurement' * obser);
x2_alpha      = gama_2(1,1)/length_col * ...
    trace( (measurement' * measurement * var_noise + gama_2(1,1)*eye(length_col)) \ eye(length_col) );
x2_eta        = gama_2(1,1) / x2_alpha;
gama_1(1,2)   = x2_eta - gama_2(1,1);
r1(:,2)       = x2_mean(:,1) * x2_eta / gama_1(1,2);

%% ================== VAMP iteration ==================
T_NUM   = 50;
nmse_it = nan(T_NUM,1);
 
nmse_it(1) = norm(x2_mean(:,1) - Symbol_X,'fro')^2 / norm(Symbol_X,'fro')^2;

for t = 2:T_NUM
    % -------- Denoiser / Module A (steps 4-6) ----------
    x1_eta(1,t)   = (gama_1(1,t) + 1/pri_var);
    x1_mean(:,t)  = (gama_1(1,t)*r1(:,t) + pri_mean/pri_var) / x1_eta(1,t);

    gama_2(1,t)   = x1_eta(1,t) - gama_1(1,t);
    r2(:,t)       = (x1_mean(:,t)*x1_eta(1,t) - r1(:,t)*gama_1(1,t)) / gama_2(1,t);

    % -------- Linear module / Module B (steps 8-10) ----
    D = pinv( (1/var_noise) * (S'*S) + gama_2(1,t)*eye(length_col) );    % = (σ0^{-2}S^HS + γ2 I)^{-1}
    x2_mean(:,t) = V * ( D * ( y_flag + gama_2(1,t) * (V' * r2(:,t)) ) );

    % Onsager 系数 beta_p = (1/N) * sum_r γ2 * D_r
    x2_eta_inv(1,t) = (gama_2(1,t) * sum(diag(D))) / length_col;

    % 更新 r1, γ1（steps 11-12）
    r1(:,t+1)   = ( x2_mean(:,t) - x2_eta_inv(1,t)*r2(:,t) ) / (1 - x2_eta_inv(1,t));
    gama_1(1,t+1)= gama_2(1,t) * (1 - x2_eta_inv(1,t)) / max(x2_eta_inv(1,t),eps);

    % -------- 记录 NMSE（用 x2_mean 作为当前估计） ----------
    post_var  = 1/(1/pri_var + gama_1(1,t+1));
    post_mean = post_var*(pri_mean/pri_var + r1(:,t+1)*gama_1(1,t+1));
    nmse_it(t) = norm(post_mean - Symbol_X,'fro')^2 / norm(Symbol_X,'fro')^2; 
end

%% ================== State Evolution (Gaussian prior + AWGN) ================== 
s  = diag(S);                      % singular values {s_i}
s2 = s.^2;
R  = numel(s2);                    % rank(A) for Gaussian A (a.s. = min(M,N))
Z  = max(N_t - R, 0);              % #zero singular values (nullspace)
N  = N_t;                          % dimension
gamma_w = 1/var_noise;             % noise precision
 
if strcmpi(cfg.sigModel,'gaussian')
    var_x = pri_var;
else
    var_x = var(Symbol_X(:));
end
 
A1_fun  = @(g1) (var_x.*g1) ./ (1 + var_x.*g1);                           % α1 = var_x*γ1/(1+var_x*γ1)
E1_fun  = @(g1,t1) (A1_fun(g1)-1).^2*var_x + (A1_fun(g1)).^2 .* t1;       % 𝓔1
 
alpha2_fun = @(g2) ( Z + sum( g2 ./ (gamma_w*s2 + g2) ) ) / N;
E2_fun     = @(g2,t2) (1/N)*( ...
                     gamma_w * sum( s2 ./ (gamma_w*s2 + g2).^2 ) + ...              % = γ_w^2 σ_w^2 Σ s_i^2/(γ_w s_i^2+g2)^2
                     (g2.^2 .* t2) .* ( Z./(g2.^2) + sum( 1 ./ (gamma_w*s2 + g2).^2 ) ) );
 
K_SE   = T_NUM;
tau1   = zeros(K_SE,1);
nmse_se= zeros(K_SE,1);
alpha1_hist = zeros(K_SE,1); alpha2_hist = zeros(K_SE,1);
E1_hist = zeros(K_SE,1);     E2_hist = zeros(K_SE,1);

tau1(1) = var_x;                   % τ1,0
gamma1   = 1/var_x;                % γ1,0
epsd     = 1e-18;

for k = 1:K_SE
    % ---- Module A ----  (32: α1, η1, γ2, τ2)
    alpha1 = A1_fun(gamma1);                                   % α1 = A1(γ1,τ1)
    E1     = E1_fun(gamma1, tau1(k));                          % 𝓔1(γ1,τ1)
    tau2   = max( (E1 - alpha1^2 * tau1(k)) / max((1 - alpha1)^2, epsd), 0 );
    eta1   = gamma1 / max(alpha1, epsd);
    gamma2 = max(eta1 - gamma1, epsd);

    % ---- Module B ----  (32–33: α2, η2; 33: 𝓔2; 32: τ1⁺, γ1⁺)
    alpha2 = alpha2_fun(gamma2);
    E2     = E2_fun(gamma2, tau2);
    tau1n  = max( (E2 - alpha2^2 * tau2) / max((1 - alpha2)^2, epsd), 0 );
    eta2   = gamma2 / max(alpha2, epsd);
    gamma1 = max(eta2 - gamma2, epsd);
 
    E1_hist(k)=E1; E2_hist(k)=E2; alpha1_hist(k)=alpha1; alpha2_hist(k)=alpha2;
    nmse_se(k) = tau1(k) / max(var_x,epsd);
    if k < K_SE, tau1(k+1) = tau1n; end
end

nmse_se = nmse_se(1:numel(nmse_it));

%% ================== Plot: NMSE vs Iterations (VAMP vs SE) ==================
figure; hold on; grid on;
semilogy(1:numel(nmse_it), nmse_it, 'o-','LineWidth',1.6);
semilogy(1:numel(nmse_se), nmse_se, 's--','LineWidth',1.6);
xlabel('Iteration'); ylabel('NMSE');
title(sprintf('VAMP vs. SE  (N_r=%d, N_t=%d, SNR=%.1f dB, \\alpha=%g)', ...
      N_r, N_t, SNR_dB, cfg.alpha));
legend('VAMP (simulation)','SE (prediction)','Location','best');

%% ================= Helpers =================
function X = make_signal(cfg, N_t, L)
    switch lower(cfg.sigModel)
        case 'gmm'
            X = zeros(N_t, L);
            mask = rand(N_t, L) < cfg.pi_mix(2);
            X(mask)  = cfg.sigma_mix(2) * randn(nnz(mask),  1);
            X(~mask) = cfg.sigma_mix(1) * randn(nnz(~mask), 1);
        case 'gaussian'
            X = randn(N_t, L);               % zero-mean Gaussian
        case 'noprior'
            X = 3 + 10*randn(N_t, L) ;           
        otherwise
            error('Unknown cfg.sigModel');
    end
end

function N = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta)
    gamma = (var_noise/2)^(1/cfg.alpha);            % ensures 2*gamma^2=var_noise if alpha=2
    N = sAS(cfg.alpha, gamma, N_r, L, delta);       % i.i.d.
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
            C = V * diag(sqrt(d)) * V.';  % Rn^{1/2}
            N = C * N;
            N = N * (sqrt(var_noise) / std(N(:)));
        else 
            for r = 1:N_r
                N(r,:) = filter(cfg.h, 1, N(r,:));
            end
            if abs(cfg.alpha - 2) < 1e-12 
                for r = 1:N_r
                    vr = var(N(r,:));
                    if vr > 0, N(r,:) = N(r,:) * sqrt(var_noise / vr); end
                end
            else 
                N = N * (sqrt(var_noise) / std(N(:)));
            end
        end
      end
end

function X = sAS(alpha, gamma, m, n, delta)
    if alpha<=0 || alpha>2, error('alpha in (0,2]'); end
    if nargin<5, delta=0; end
    if abs(alpha-2)<1e-12
        X = delta + sqrt(2)*gamma*randn(m,n); % Gaussian: N(delta, 2*gamma^2)
        return;
    end
    U = pi*(rand(m,n)-0.5);
    W = exprnd(1, m, n);
    X0 = gamma .* (sin(alpha.*U) ./ (cos(U).^(1/alpha))) .* ...
         ((cos((1-alpha).*U) ./ W).^((1-alpha)/alpha));
    X = X0 + delta;
end
