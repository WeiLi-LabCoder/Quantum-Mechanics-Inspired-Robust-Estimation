%% ================= NMSE ================= 

clear; clc; rng(1);
% ---------- Dimensions ----------
N_r = 16; N_t = 8; L = 1;
SNR_Range = -5:5:20;
FrameAll = 5000;
% ---------- Channel configuration ----------
var_channel = 5;
cfg.trueVarScale  = 1.0;    
cfg.forceMeanBias = 0.0;    

cfg.fadingModel     = 'rayleigh';   % 'rayleigh' | 'rician'
cfg.K_rician        = 5;
cfg.useCorrChan     = false;
cfg.rho_r           = 0.7;
cfg.rho_t           = 0.5;

cfg.useULAsteering = false; 
cfg.d_over_lambda  = 0.5; 
cfg.AoA            = [];       
cfg.AoD            = []; 

cfg.useImperfectCSI = false;
cfg.csiErrVar       = var_channel*1e-2;

% ---------- Noise ----------
cfg.alpha      = 1;           % 0<alpha<=2; alpha=2 -> Gaussian
cfg.isComplex = false; 
  
% ---------- Signal model ----------
% 'known' | 'partial' | 'absent'
cfg.sigModel   = 'known';      % 'partial' | 'known' | 'absent'
cfg.pi_mix     = [0.5 0.5];  % GMM mixing weights (sum=1)
cfg.sigma_mix  = [1 10];     % GMM stds, zero-mean components

% ---------- For α-stable noise ----------
P_sig = 1;
delta = 0;

% ---------- quantum/Hamiltonian items ----------
hbar = 1; mass = 1;
const_Hamil = hbar^2 / (2*mass);
Hamil_P_base = Hamiltonian_Original(N_r, L, const_Hamil);

% ---------- Robust IRLS settings ----------
rob.maxIter = 15;
rob.tol     = 1e-7;
rob.epsW    = 1e-8;
rob.huber_c = 1.345;   % standard Huber constant (multiplies robust scale)

 
% --- NEW: robust baselines (on Y and on Ytilde) ---
err_x_LS             = zeros(1, numel(SNR_Range));
err_x_LAD            = zeros(1, numel(SNR_Range));
err_x_Huber          = zeros(1, numel(SNR_Range)); 
 
for iter = 1:numel(SNR_Range)
    SNR_dB    = SNR_Range(iter);
    var_noise = P_sig / 10^(SNR_dB/10); 

    frame_valid_QD = 0;
    frame_valid_PP = 0;

    for frame = 1:FrameAll
        % ===== Noise =====
        noise = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta);
        gamma = (var_noise/2)^(1/cfg.alpha);
        % ===== Tx Signal ===== 
        Symbol_X = make_signal(cfg, N_t, L);
        % ===== Channel =====
        Channel_H = make_channel_only(cfg, N_r, N_t, var_channel);
        H_used = Channel_H;
        if isfield(cfg,'useImperfectCSI') && cfg.useImperfectCSI
            E = (randn(N_r,N_t)+1j*randn(N_r,N_t))/sqrt(2);
            H_used = Channel_H + sqrt(cfg.csiErrVar)*E;
        end

        % ===== I/O =====
        Chan_Symbol = Channel_H * Symbol_X;
        Rec_Y       = Chan_Symbol + noise;
        y_vec = Rec_Y(:);
 

        % ===== LS baselines =====
        Hpinv  = pinv(H_used);
        Ht_LS = Hpinv * Rec_Y;  
        err_x_LS(iter) = err_x_LS(iter) + nmse(Ht_LS,Symbol_X);
        % ===== Robust baselines ===== 
        Xhat_LAD = zeros(N_t,L);
        Xhat_Huber = zeros(N_t,L); 

        for r = 1:L
            ycol  = Rec_Y(:,r); 

            Xhat_LAD(:,r)      = robust_lad_irls(H_used,ycol,rob);
            Xhat_Huber(:,r)    = robust_huber_irls(H_used,ycol,rob);
 
        end

        err_x_LAD(iter)      = err_x_LAD(iter) + nmse(Xhat_LAD,Symbol_X);
        err_x_Huber(iter)    = err_x_Huber(iter) + nmse(Xhat_Huber,Symbol_X);
 
 
    end

    % ===== Average NMSE ===== 
    err_x_LS(iter)          = err_x_LS(iter)          / FrameAll;
    err_x_LAD(iter)         = err_x_LAD(iter)         / FrameAll;
    err_x_Huber(iter)       = err_x_Huber(iter)       / FrameAll; 
 
 
end

% ===== Titles =====
noiseTypeStr = sprintf('$\\alpha$-stable noise ($\\alpha=%g$)', cfg.alpha); 
 
chanStr = cfg.fadingModel;
if cfg.useImperfectCSI, chanStr = [chanStr, ', imperfect CSI']; else, chanStr = [chanStr, ', perfect CSI']; end
if cfg.useCorrChan, chanStr = [chanStr, ', correlated']; else, chanStr = [chanStr, ', i.i.d.']; end

priorStr = sprintf('priorCase=%s', cfg.sigModel);

%% ================= Figure: NMSE in X =================
figure 
semilogy(SNR_Range, err_x_LS, '-o','Color', [0.00, 0.45, 0.74], 'LineWidth', 2);
hold on 
semilogy(SNR_Range, err_x_LAD, '--o', 'LineWidth', 1.8); 
semilogy(SNR_Range, err_x_Huber, '--^', 'LineWidth', 1.8); 

grid on;
xlabel('SNR (dB)','Interpreter','latex');
ylabel('NMSE in $\mathbf{X}$','Interpreter','latex');
legend({ 'LS on $\mathbf{Y}$','LAD($\ell_1$) on $\mathbf{Y}$','Huber on $\mathbf{Y}$'}, ... 
        'Interpreter','latex','Location','best');
title(sprintf('%s, %s, %s, %s', noiseTypeStr,  chanStr, priorStr),'Interpreter','latex');
 

%% ================= Helpers =================
 
 

function v = nmse(A,B)
    v = norm(A - B, 'fro')^2 / max(norm(B, 'fro')^2, eps);
end

function v = nmse_vec(x_vec, B)
    v = norm(x_vec - B(:))^2 / max(norm(B(:))^2, eps);
end

 
 %% ===== Robust baseline: LAD via IRLS =====
 function x = robust_lad_irls(H, y, rob)
% min ||y - Hx||_1  (complex-safe using |r|)

    x = pinv(H) * y; 

    for it = 1:rob.maxIter
        r = y - H*x;

        % LAD weights: w_i = 1/|r_i|
        w = 1 ./ max(abs(r), rob.epsW);

        % use sqrt(w) for proper WLS
        sw = sqrt(w);
        Hw = H .* sw;
        yw = y .* sw;

        % small Tikhonov to avoid singularity (important!)
        lam = 1e-10;
        x_new = (Hw' * Hw + lam*eye(size(H,2))) \ (Hw' * yw);

        if norm(x_new-x)^2 <= rob.tol * max(norm(x)^2,1)
            x = x_new; return;
        end
        x = x_new;
    end
end


function x = robust_huber_irls(H, y, rob)
% Huber IRLS (complex-safe using |r|)

    x = pinv(H) * y; 

    for it = 1:rob.maxIter
        r = y - H*x;
        ar = abs(r);

        % robust scale for complex magnitude (simple & stable)
        s = median(ar);
        s = max(s, 1e-12);

        c = rob.huber_c * s;

        % Huber weights:
        % w = 1                      if |r| <= c
        % w = c/|r|                  if |r| >  c
        w = ones(size(r));
        idx = ar > c;
        w(idx) = c ./ max(ar(idx), rob.epsW);

        sw = sqrt(w);
        Hw = H .* sw;
        yw = y .* sw;

        lam = 1e-10;
        x_new = (Hw' * Hw + lam*eye(size(H,2))) \ (Hw' * yw);

        if norm(x_new-x)^2 <= rob.tol * max(norm(x)^2,1)
            x = x_new; return;
        end
        x = x_new;
    end
end

function X = make_signal(cfg, N_t, L)
    switch lower(cfg.sigModel)
        case 'partial'
            X = zeros(N_t, L);
            mask = rand(N_t, L) < cfg.pi_mix(2);
            X(mask)  = cfg.sigma_mix(2) * randn(nnz(mask),  1);
            X(~mask) = cfg.sigma_mix(1) * randn(nnz(~mask), 1);
        case 'known'
            X = randn(N_t, L);              
        case 'absent'
            X = 3 + 10*randn(N_t, L) ;           
        otherwise
            error('Unknown cfg.sigModel');
    end  
end
 