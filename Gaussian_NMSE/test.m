%% ================= NMSE ================= 

clear; clc; rng(1);
% ---------- Dimensions ----------
N_r = 16; N_t = 16; L = 1;
SNR_Range = -5:5:20;
FrameAll = 10;
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
cfg.csiErrVar       = 1e-2;

% ---------- Noise ----------
cfg.alpha      = 2;           % 0<alpha<=2; alpha=2 -> Gaussian
 
  
% ---------- Signal model ----------
% 'known' | 'partial' | 'absent'
cfg.sigModel   = 'absent';      % 'partial' | 'known' | 'absent'
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

% ---------- Collectors: NMSE ----------
err_y_NL = zeros(1, numel(SNR_Range));
err_y    = zeros(1, numel(SNR_Range));

err_x_LS             = zeros(1, numel(SNR_Range));
err_x_LS_QD          = zeros(1, numel(SNR_Range));
err_h_NL_LS_QD       = zeros(1, numel(SNR_Range));

err_x_VAMP_ori       = zeros(1, numel(SNR_Range));
err_x_VAMP_QD        = zeros(1, numel(SNR_Range));
err_x_VAMP_PlugPlay  = zeros(1, numel(SNR_Range));

% --- NEW: robust baselines (on Y and on Ytilde) ---
err_x_LAD            = zeros(1, numel(SNR_Range));
err_x_Huber          = zeros(1, numel(SNR_Range));
err_x_LAD_QD         = zeros(1, numel(SNR_Range));
err_x_Huber_QD       = zeros(1, numel(SNR_Range));
 
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
        Symbol_X = make_signal(cfg, N_t, L);
        switch lower(cfg.sigModel)
            case 'known'
                mu0  = 0;
                var0 = 1;
        
            case 'partial'
                mu0  = 0;
                var0 = cfg.pi_mix(1)*cfg.sigma_mix(1)^2 + ...
                       cfg.pi_mix(2)*cfg.sigma_mix(2)^2;
        
            case {'absent','noprior','unknown'}
                mu0  = 0;
                var0 = 1e6;
        
            otherwise
                error('Unknown cfg.sigModel');
        end
        % ===== Channel =====
        Channel_H = make_channel_only(cfg, N_r, N_t, var_channel);
        H_used = Channel_H;
        if isfield(cfg,'useImperfectCSI') && cfg.useImperfectCSI
            if isfield(cfg,'isComplex') && cfg.isComplex
                E = (randn(N_r,N_t)+1j*randn(N_r,N_t))/sqrt(2);
            else
                E = randn(N_r,N_t);
            end
            H_used = Channel_H + sqrt(cfg.csiErrVar)*E;
        end

        % ===== I/O =====
        Chan_Symbol = Channel_H * Symbol_X;
        Rec_Y       = Chan_Symbol + noise;
        y_vec = Rec_Y(:);

        Chan_Symbol_used = H_used * Symbol_X;

        % ===== Denoise Y (QD) =====
        % [Rec_Y_NL_vec, tau, U_P] = DenoisedSignal_QD_trained(Hamil_P_base, Rec_vec_y,beta_opt);
        [y_denoised_vec, tau, U_P] = DenoisedSignal_QD(Hamil_P_base, y_vec, Chan_Symbol_used);
 
        Yt_denoised = reshape(y_denoised_vec,N_r,L);

        % ===== Y NMSE =====
        err_y(iter) = err_y(iter) + nmse(Rec_Y,Chan_Symbol);
        err_y_NL(iter) = err_y_NL(iter) + nmse(Yt_denoised,Chan_Symbol);

        % ===== LS baselines =====
        Hpinv  = pinv(H_used);
        Ht_LS = Hpinv * Rec_Y;
        Ht_LS_QD = Hpinv * Yt_denoised;

        err_x_LS(iter) = err_x_LS(iter) + nmse(Ht_LS,Symbol_X);
        err_x_LS_QD(iter) = err_x_LS_QD(iter) + nmse(Ht_LS_QD,Symbol_X);


        % ===== Robust baselines ===== 
        Xhat_LAD = zeros(N_t,L);
        Xhat_Huber = zeros(N_t,L);
        Xhat_LAD_QD = zeros(N_t,L);
        Xhat_Huber_QD = zeros(N_t,L);

        for r = 1:L
            ycol  = Rec_Y(:,r);
            ycolQ = Yt_denoised(:,r);

            Xhat_LAD(:,r)      = robust_lad_irls(H_used,ycol,rob);
            Xhat_Huber(:,r)    = robust_huber_irls(H_used,ycol,rob);

            Xhat_LAD_QD(:,r)   = robust_lad_irls(H_used,ycolQ,rob);
            Xhat_Huber_QD(:,r) = robust_huber_irls(H_used,ycolQ,rob);
        end

        err_x_LAD(iter)      = err_x_LAD(iter) + nmse(Xhat_LAD,Symbol_X);
        err_x_Huber(iter)    = err_x_Huber(iter) + nmse(Xhat_Huber,Symbol_X);
        err_x_LAD_QD(iter)   = err_x_LAD_QD(iter) + nmse(Xhat_LAD_QD,Symbol_X);
        err_x_Huber_QD(iter) = err_x_Huber_QD(iter) + nmse(Xhat_Huber_QD,Symbol_X);
        
        
        A_base = kron(eye(L),H_used);
        mat_A = U_P * diag(tau) * U_P' * A_base; 
        % ===== VAMP variants =====
        [est_X_VAMP_ori,~] = Recovery_VAMP(y_vec, A_base, mu0, var0, gamma);
        if all(isfinite(est_X_VAMP_ori(:)))
            err_x_VAMP_ori(iter) = err_x_VAMP_ori(iter) + nmse_vec(est_X_VAMP_ori, Symbol_X);
        end

        [est_x_VAMP_QD,~] = Recovery_VAMP(y_denoised_vec, mat_A,  mu0, var0, gamma);
        if all(isfinite(est_x_VAMP_QD(:)))
            err_x_VAMP_QD(iter) = err_x_VAMP_QD(iter) + nmse_vec(est_x_VAMP_QD, Symbol_X);
            frame_valid_QD = frame_valid_QD + 1;
        end

        est_x_PP = QD_VAMP(y_denoised_vec, mat_A,  mu0, var0, gamma, Symbol_X(:), const_Hamil);
        if all(isfinite(est_x_PP(:)))
            err_x_VAMP_PlugPlay(iter) = err_x_VAMP_PlugPlay(iter) + nmse_vec(est_x_PP, Symbol_X);
            frame_valid_PP = frame_valid_PP + 1;
        end
 
    end

    % ===== Average NMSE =====
    err_y_NL(iter) = err_y_NL(iter) / FrameAll;
    err_y(iter)    = err_y(iter)    / FrameAll;

    err_x_LS(iter)          = err_x_LS(iter)          / FrameAll;
    err_x_LS_QD(iter)       = err_x_LS_QD(iter)       / FrameAll; 
    err_x_VAMP_ori(iter)    = err_x_VAMP_ori(iter)    / FrameAll;

    err_x_LAD(iter)         = err_x_LAD(iter)         / FrameAll;
    err_x_Huber(iter)       = err_x_Huber(iter)       / FrameAll;
    err_x_LAD_QD(iter)      = err_x_LAD_QD(iter)      / FrameAll;
    err_x_Huber_QD(iter)    = err_x_Huber_QD(iter)    / FrameAll;

    if frame_valid_QD > 0
        err_x_VAMP_QD(iter) = err_x_VAMP_QD(iter) / frame_valid_QD;
    else
        err_x_VAMP_QD(iter) = NaN;
        warning('All VAMP-QD frames invalid at SNR=%d dB', SNR_Range(iter));
    end

    if frame_valid_PP > 0
        err_x_VAMP_PlugPlay(iter) = err_x_VAMP_PlugPlay(iter) / frame_valid_PP;
    else
        err_x_VAMP_PlugPlay(iter) = NaN;
        warning('All VAMP-PP frames invalid at SNR=%d dB', SNR_Range(iter));
    end
 
end

% ===== Titles =====
noiseTypeStr = sprintf('$\\alpha$-stable noise ($\\alpha=%g$)', cfg.alpha); 
 
chanStr = cfg.fadingModel;
if cfg.useImperfectCSI, chanStr = [chanStr, ', imperfect CSI']; else, chanStr = [chanStr, ', perfect CSI']; end
if cfg.useCorrChan, chanStr = [chanStr, ', correlated']; else, chanStr = [chanStr, ', i.i.d.']; end

priorStr = sprintf('priorCase=%s', cfg.sigModel);

%% ================= Figure: NMSE in X =================
figure
semilogy(SNR_Range, err_x_LS, '-o','Color', [0.00, 0.45, 0.74], 'LineWidth', 2); hold on
semilogy(SNR_Range, err_x_VAMP_ori, '-^','Color',[0.35, 0.80, 0.70],  'LineWidth', 2);
semilogy(SNR_Range, err_x_LS_QD, '-s','Color', [0.47, 0.67, 0.19],  'LineWidth', 2); 
semilogy(SNR_Range, err_x_VAMP_QD, '->', 'Color', [0.93, 0.69, 0.13],'LineWidth', 2);
semilogy(SNR_Range, err_x_VAMP_PlugPlay, '-<', 'Color', [0.85, 0.33, 0.10], 'LineWidth', 2);

% NEW robust curves
semilogy(SNR_Range, err_x_LAD, '--o', 'LineWidth', 1.8);
semilogy(SNR_Range, err_x_Huber, '--^', 'LineWidth', 1.8);
semilogy(SNR_Range, err_x_LAD_QD, '--s', 'LineWidth', 1.8);
semilogy(SNR_Range, err_x_Huber_QD, '--d', 'LineWidth', 1.8);

grid on;
xlabel('SNR (dB)','Interpreter','latex');
ylabel('NMSE in $\mathbf{X}$','Interpreter','latex');
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

 
 %% ===== Robust baseline: LAD via IRLS =====
function x = robust_lad_irls(H, y, rob)
% min ||y - Hx||_1  (complex-safe using |r|)

    % x = pinv(H) * y;
    x = zeros(size(H,2),1);

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

    % x = pinv(H) * y;
    x = zeros(size(H,2),1);

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

% function X = make_signal(cfg, N_t, L)
%     switch lower(cfg.sigModel)
%         case 'partial'
%             X = zeros(N_t, L);
%             mask = rand(N_t, L) < cfg.pi_mix(2);
%             X(mask)  = cfg.sigma_mix(2) * randn(nnz(mask),  1);
%             X(~mask) = cfg.sigma_mix(1) * randn(nnz(~mask), 1);
%         case 'known'
%             X = randn(N_t, L);              
%         case 'absent'
%             X = 3 + 10*randn(N_t, L) ;           
%         otherwise
%             error('Unknown cfg.sigModel');
%     end  
% end

function X = make_signal(cfg, N_t, L)
    switch lower(cfg.sigModel)
        case 'partial'
            X = zeros(N_t, L);
            mask = rand(N_t, L) < cfg.pi_mix(2);
            X(mask)  = cfg.sigma_mix(2) * randn(nnz(mask),  1);
            X(~mask) = cfg.sigma_mix(1) * randn(nnz(~mask), 1);

            % analytical normalization
            sig2 = cfg.pi_mix(1)*cfg.sigma_mix(1)^2 + ...
                   cfg.pi_mix(2)*cfg.sigma_mix(2)^2;
            X = X / sqrt(sig2);

        case 'known'
            X = randn(N_t, L);

        case 'absent'
            X = (3 + 10*randn(N_t, L)) / sqrt(109);

        otherwise
            error('Unknown cfg.sigModel');
    end
end