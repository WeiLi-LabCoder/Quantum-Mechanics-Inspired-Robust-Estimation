%% ================= NMSE ================= 

clear; clc; rng(1);
% ---------- Dimensions ----------
N_r = 1; N_t = 8; L = 16;
SNR_Range = -5:5:20;
FrameAll = 3000;
% ---------- Channel configuration ----------
var_channel = 5;
cfg.trueVarScale  = 1.0;    
cfg.forceMeanBias = 0.0;    

cfg.fadingModel     = 'rician';   % 'rayleigh' | 'rician'
cfg.K_rician        = 5;
cfg.useCorrChan     = false;
cfg.rho_r           = 0.7;
cfg.rho_t           = 0.5;

cfg.useULAsteering = false; 
cfg.d_over_lambda  = 0.5; 
cfg.AoA            = [];       
cfg.AoD            = []; 

cfg.useImperfectCSI = false;
cfg.csiErrVar       = 1e-2;

% ---------- Noise ----------
cfg.alpha      = 1;           % 0<alpha<=2; alpha=2 -> Gaussian
 

% Signal model: 
cfg.mod   = '16QAM'; 
cfg.Mconst = get_mod_order(cfg.mod);
 
  
% ---------- PRIOR CASE (statistical prior: mean/variance) ----------
% 'known' | 'partial' | 'absent'
cfg.priorCase = 'known';
cfg.partialMode = 'var_mismatch';   % 'var_mismatch' | 'mean_mismatch' | 'both'
cfg.varMismatchFactor = 2.0;
cfg.meanMismatchBias  = 0.3;
cfg.absentMode = 'default01';       % 'default01' | 'weak'
cfg.weakPriorVar = 1e3;

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

% ---------- Collectors: NMSE ----------
err_y_NL = zeros(1, numel(SNR_Range));
err_y    = zeros(1, numel(SNR_Range));

err_h_LS             = zeros(1, numel(SNR_Range));
err_h_LS_QD          = zeros(1, numel(SNR_Range));
err_h_NL_LS_QD       = zeros(1, numel(SNR_Range));

err_h_VAMP_ori       = zeros(1, numel(SNR_Range));
err_h_VAMP_QD        = zeros(1, numel(SNR_Range));
err_h_VAMP_PlugPlay  = zeros(1, numel(SNR_Range));

% --- NEW: robust baselines (on Y and on Ytilde) ---
err_h_LAD            = zeros(1, numel(SNR_Range));
err_h_Huber          = zeros(1, numel(SNR_Range));
err_h_LAD_QD         = zeros(1, numel(SNR_Range));
err_h_Huber_QD       = zeros(1, numel(SNR_Range));
 
for iter = 1:numel(SNR_Range)
    SNR_dB    = SNR_Range(iter);
    var_noise = P_sig / 10^(SNR_dB/10);
    % beta_opt = beta_SNR(:, iter);

    frame_valid_QD = 0;
    frame_valid_PP = 0;

    for frame = 1:FrameAll
        % ===== Noise =====
        noise = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta);
        gamma = (var_noise/2)^(1/cfg.alpha);
        % ===== Tx Signal ===== 
        Symbol_X = qammod(randi([0 cfg.Mconst-1],N_t,L),cfg.Mconst,'UnitAveragePower',true);
        % ===== Channel =====
        [Channel_H, stats] = make_channel(cfg, N_r, N_t, var_channel);
        % ===== Statistical prior (mean/var) for VAMP family =====
        [pri_mean, pri_var] = choose_stat_prior_H(cfg, stats);

        % ===== I/O =====
        Chan_Symbol = Channel_H * Symbol_X;
        Rec_Y       = Chan_Symbol + noise;

        % ===== Transpose ========
        Yt = Rec_Y';
        Xt = Symbol_X';
        Ht_true = Channel_H';
        y_vec = Yt(:);

        % ===== Denoise Y (QD) =====
        % [Rec_Y_NL_vec, tau, U_P] = DenoisedSignal_QD_trained(Hamil_P_base, Rec_vec_y,beta_opt);
        [y_denoised_vec, tau, U_P] = DenoisedSignal_QD(Hamil_P_base, y_vec, Chan_Symbol');
 
        Yt_denoised = reshape(y_denoised_vec,L,N_r);

        % ===== Y NMSE =====
        err_y(iter) = err_y(iter) + nmse(Yt,Chan_Symbol');
        err_y_NL(iter) = err_y_NL(iter) + nmse(Yt_denoised,Chan_Symbol');

        % ===== LS baselines =====
        Xpinv  = pinv(Xt);
        Ht_LS = Xpinv * Yt;
        Ht_LS_QD = Xpinv * Yt_denoised;

        err_h_LS(iter) = err_h_LS(iter) + nmse(Ht_LS,Ht_true);
        err_h_LS_QD(iter) = err_h_LS_QD(iter) + nmse(Ht_LS_QD,Ht_true);


        % ===== Robust baselines ===== 
        Hhat_LAD = zeros(N_t,N_r);
        Hhat_Huber = zeros(N_t,N_r);
        Hhat_LAD_QD = zeros(N_t,N_r);
        Hhat_Huber_QD = zeros(N_t,N_r);

        for r = 1:N_r
            ycol  = Yt(:,r);
            ycolQ = Yt_denoised(:,r);

            Hhat_LAD(:,r)      = robust_lad_irls(Xt,ycol,rob);
            Hhat_Huber(:,r)    = robust_huber_irls(Xt,ycol,rob);

            Hhat_LAD_QD(:,r)   = robust_lad_irls(Xt,ycolQ,rob);
            Hhat_Huber_QD(:,r) = robust_huber_irls(Xt,ycolQ,rob);
        end

        err_h_LAD(iter)      = err_h_LAD(iter) + nmse(Hhat_LAD,Ht_true);
        err_h_Huber(iter)    = err_h_Huber(iter) + nmse(Hhat_Huber,Ht_true);
        err_h_LAD_QD(iter)   = err_h_LAD_QD(iter) + nmse(Hhat_LAD_QD,Ht_true);
        err_h_Huber_QD(iter) = err_h_Huber_QD(iter) + nmse(Hhat_Huber_QD,Ht_true);
        
        
        A_base = kron(eye(N_r),Xt);
        mat_A = U_P * diag(tau) * U_P' * A_base; 
        % ===== VAMP variants =====
        [est_h_VAMP_ori,~] = Recovery_VAMP(y_vec, A_base, pri_mean, pri_var, gamma);
        if all(isfinite(est_h_VAMP_ori(:)))
            err_h_VAMP_ori(iter) = err_h_VAMP_ori(iter) + nmse_vec(est_h_VAMP_ori, Ht_true);
        end

        [est_h_VAMP_QD,~] = Recovery_VAMP(y_denoised_vec, mat_A, pri_mean, pri_var, gamma);
        if all(isfinite(est_h_VAMP_QD(:)))
            err_h_VAMP_QD(iter) = err_h_VAMP_QD(iter) + nmse_vec(est_h_VAMP_QD, Ht_true);
            frame_valid_QD = frame_valid_QD + 1;
        end

        est_h_PP = QD_VAMP(y_denoised_vec, mat_A, pri_mean, pri_var, gamma, Ht_true(:), const_Hamil);
        if all(isfinite(est_h_PP(:)))
            err_h_VAMP_PlugPlay(iter) = err_h_VAMP_PlugPlay(iter) + nmse_vec(est_h_PP, Ht_true);
            frame_valid_PP = frame_valid_PP + 1;
        end

        % ===== Nonlinear LS on Y_tilde =====
        % Hhat_NL = QD_NL_LS(Xt, U_P, tau, Yt_denoised, Ht_true);
        % Hhat_NL = reshape(Hhat_NL, N_t, N_r);
        % err_h_NL_LS_QD(iter) = err_h_NL_LS_QD(iter) + nmse(Hhat_NL, Ht_true);
 
        
    end

    % ===== Average NMSE =====
    err_y_NL(iter) = err_y_NL(iter) / FrameAll;
    err_y(iter)    = err_y(iter)    / FrameAll;

    err_h_LS(iter)          = err_h_LS(iter)          / FrameAll;
    err_h_LS_QD(iter)       = err_h_LS_QD(iter)       / FrameAll;
    % err_h_NL_LS_QD(iter)    = err_h_NL_LS_QD(iter)    / FrameAll;
    err_h_VAMP_ori(iter)    = err_h_VAMP_ori(iter)    / FrameAll;

    err_h_LAD(iter)         = err_h_LAD(iter)         / FrameAll;
    err_h_Huber(iter)       = err_h_Huber(iter)       / FrameAll;
    err_h_LAD_QD(iter)      = err_h_LAD_QD(iter)      / FrameAll;
    err_h_Huber_QD(iter)    = err_h_Huber_QD(iter)    / FrameAll;

    if frame_valid_QD > 0
        err_h_VAMP_QD(iter) = err_h_VAMP_QD(iter) / frame_valid_QD;
    else
        err_h_VAMP_QD(iter) = NaN;
        warning('All VAMP-QD frames invalid at SNR=%d dB', SNR_Range(iter));
    end

    if frame_valid_PP > 0
        err_h_VAMP_PlugPlay(iter) = err_h_VAMP_PlugPlay(iter) / frame_valid_PP;
    else
        err_h_VAMP_PlugPlay(iter) = NaN;
        warning('All VAMP-PP frames invalid at SNR=%d dB', SNR_Range(iter));
    end
 
end

% ===== Titles =====
noiseTypeStr = sprintf('$\\alpha$-stable noise ($\\alpha=%g$)', cfg.alpha); 
 
chanStr = cfg.fadingModel;
if cfg.useImperfectCSI, chanStr = [chanStr, ', imperfect CSI']; else, chanStr = [chanStr, ', perfect CSI']; end
if cfg.useCorrChan, chanStr = [chanStr, ', correlated']; else, chanStr = [chanStr, ', i.i.d.']; end

priorStr = sprintf('priorCase=%s', cfg.priorCase);

%% ================= Figure: NMSE in H =================
figure
semilogy(SNR_Range, err_h_LS, '-o','Color', [0.00, 0.45, 0.74], 'LineWidth', 2); hold on
semilogy(SNR_Range, err_h_VAMP_ori, '-^','Color',[0.35, 0.80, 0.70],  'LineWidth', 2);
semilogy(SNR_Range, err_h_LS_QD, '-s','Color', [0.47, 0.67, 0.19],  'LineWidth', 2);
% semilogy(SNR_Range, err_h_NL_LS_QD, '-d', 'Color', [0.49, 0.18, 0.56],  'LineWidth', 2);
semilogy(SNR_Range, err_h_VAMP_QD, '->', 'Color', [0.93, 0.69, 0.13],'LineWidth', 2);
semilogy(SNR_Range, err_h_VAMP_PlugPlay, '-<', 'Color', [0.85, 0.33, 0.10], 'LineWidth', 2);

% NEW robust curves
semilogy(SNR_Range, err_h_LAD, '--o', 'LineWidth', 1.8);
semilogy(SNR_Range, err_h_Huber, '--^', 'LineWidth', 1.8);
semilogy(SNR_Range, err_h_LAD_QD, '--s', 'LineWidth', 1.8);
semilogy(SNR_Range, err_h_Huber_QD, '--d', 'LineWidth', 1.8);

grid on;
xlabel('SNR (dB)','Interpreter','latex');
ylabel('NMSE in $\mathbf{H}$','Interpreter','latex');
legend({'LS on $\mathbf{Y}$','VAMP on $\mathbf{Y}$', ...
        'LS on $\tilde{\mathbf{Y}}$',   ...
        'QD-VAMP on $\tilde{\mathbf{Y}}$', 'PnP-VAMP-QD on $\tilde{\mathbf{Y}}$', ...
        'LAD($\ell_1$) on $\mathbf{Y}$','Huber on $\mathbf{Y}$', ...
        'LAD($\ell_1$) on $\tilde{\mathbf{Y}}$','Huber on $\tilde{\mathbf{Y}}$'}, ...
        'Interpreter','latex','Location','best');
title(sprintf('%s, %s, %s, %s', noiseTypeStr,  chanStr, priorStr),'Interpreter','latex');
 

%% ================= Helpers =================
 
 

function v = nmse(A,B)
    v = norm(A - B, 'fro')^2 / max(norm(B, 'fro')^2, eps);
end

function v = nmse_vec(x_vec, B)
    v = norm(x_vec - B(:))^2 / max(norm(B(:))^2, eps);
end

 
% function [x_est, nmse_val] = QD_NL_LS(H, Psi, tau, Y_denoised, X_true)
%     max_iter = 800;
%     eta = 1e-3;
%     [~, L] = size(Y_denoised);
%     N_t = size(H, 2);
% 
%     y_vec = Y_denoised(:);
%     x0 = pinv(H) * Y_denoised;
%     x0 = x0(:);
% 
%     PsiT = Psi';
%     Tau  = diag(tau);
%     H_big = kron(eye(L), H);
% 
%     A = Psi * Tau * PsiT * H_big;
%     f_x = @(x) A * x;
% 
%     x = x0;
%     for iter = 1:max_iter
%         res  = y_vec - f_x(x);
%         grad = -2 * (A') * res;
%         x = x - eta * grad;
%     end
% 
%     x_est = reshape(x, N_t, L);
%     nmse_val = norm(x_est - X_true, 'fro')^2 / max(norm(X_true, 'fro')^2, eps);
% end

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