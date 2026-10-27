%% ================= Experiment: α-stable / Colored noise / Complex baseband / Fading / (Imperfect CSI) / NMSE + BER + Robust Baselines =================
% Added robust baselines:
%   (1) LAD (L1) regression via IRLS
%   (2) Huber M-estimation via IRLS
%
% Notes:
% - BER uses qammod/qamdemod (Communications Toolbox).
% - Robust LS baselines do NOT require priors (pri_mean/pri_var).
% - Complex-safe robust IRLS uses residual magnitudes |r| for weights.

clear; clc; rng(1);

% ---------- Dimensions ----------
N_r = 16; N_t = 8; L = 1;
SNR_Range = -5:5:20;
FrameAll = 3000;

% ---------- Channel power ----------
var_channel = 1;

% ---------- Config ----------
cfg.alpha      = 1;           % 0<alpha<=2; alpha=2 -> Gaussian
cfg.useColored = false;       % color the noise?
cfg.h          = [1 -1];      % FIR for colored noise
cfg.colorMode  = 'spatial';   % 'spatial' or 'temporal'

% Signal model:
% 'gmm' | 'gaussian' | 'noprior' | 'qam'
cfg.sigModel   = 'qam';       % <-- for BER experiments, use 'qam'
cfg.pi_mix     = [0.5 0.5];
cfg.sigma_mix  = [1 10];

% ---------- Wireless realism toggles ----------
cfg.isComplex       = true;
cfg.fadingModel     = 'rayleigh';   % 'rayleigh' | 'rician'
cfg.K_rician        = 5;
cfg.useCorrChan     = false;
cfg.rho_r           = 0.7;
cfg.rho_t           = 0.5;

cfg.useImperfectCSI = false;
cfg.csiErrVar       = 1e-2;

% ---------- Detection / BER ----------
cfg.mod   = '16QAM';
cfg.doBER = true;

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
rob.maxIter = 50;
rob.tol     = 1e-7;
rob.epsW    = 1e-8;
rob.huber_c = 1.345;   % standard Huber constant (multiplies robust scale)

% ---------- Collectors: NMSE ----------
err_y_NL = zeros(1, numel(SNR_Range));
err_y    = zeros(1, numel(SNR_Range));

err_x_LS             = zeros(1, numel(SNR_Range));
err_x_LS_QD          = zeros(1, numel(SNR_Range));
err_x_NL_LS_QD       = zeros(1, numel(SNR_Range));

err_x_VAMP_ori       = zeros(1, numel(SNR_Range));
err_x_VAMP_QD        = zeros(1, numel(SNR_Range));
err_x_VAMP_PlugPlay  = zeros(1, numel(SNR_Range));

% --- NEW: robust baselines (on Y and on Ytilde) ---
err_x_LAD            = zeros(1, numel(SNR_Range));
err_x_Huber          = zeros(1, numel(SNR_Range));
err_x_LAD_QD         = zeros(1, numel(SNR_Range));
err_x_Huber_QD       = zeros(1, numel(SNR_Range));

% ---------- Collectors: BER ----------
ber_LS       = zeros(1, numel(SNR_Range));
ber_VAMP_ori = zeros(1, numel(SNR_Range));
ber_LS_QD    = zeros(1, numel(SNR_Range));
ber_NL_LS_QD = zeros(1, numel(SNR_Range));
ber_VAMP_QD  = zeros(1, numel(SNR_Range));
ber_PnP      = zeros(1, numel(SNR_Range));

% --- NEW: robust BER ---
ber_LAD      = zeros(1, numel(SNR_Range));
ber_Huber    = zeros(1, numel(SNR_Range));
ber_LAD_QD   = zeros(1, numel(SNR_Range));
ber_Huber_QD = zeros(1, numel(SNR_Range));

 
for iter = 1:numel(SNR_Range)
    SNR_dB    = SNR_Range(iter);
    var_noise = P_sig / 10^(SNR_dB/10);
    % beta_opt = beta_SNR(:, iter);

    frame_valid_QD = 0;
    frame_valid_PP = 0;

    % ---------- Precompute modulation order ----------
    doBER_now = isfield(cfg,'doBER') && cfg.doBER && strcmpi(cfg.sigModel,'qam');
    if doBER_now
        Mconst = get_mod_order(cfg.mod);
    else
        Mconst = [];
    end

    % BER valid counters
    ber_valid_LS = 0; ber_valid_VAMP_ori = 0; ber_valid_LS_QD = 0;
    ber_valid_NL = 0; ber_valid_VAMP_QD = 0;  ber_valid_PnP = 0;
    ber_valid_LAD = 0; ber_valid_Huber = 0; ber_valid_LAD_QD = 0; ber_valid_Huber_QD = 0;

    for frame = 1:FrameAll
        % ===== Noise =====
        noise = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta);

        % ===== Channel =====
        Channel_H = make_channel(cfg, N_r, N_t, var_channel);

        % ===== Imperfect CSI for recovery =====
        H_used = Channel_H;
        if isfield(cfg,'useImperfectCSI') && cfg.useImperfectCSI
            if isfield(cfg,'isComplex') && cfg.isComplex
                E = (randn(N_r,N_t)+1j*randn(N_r,N_t))/sqrt(2);
            else
                E = randn(N_r,N_t);
            end
            H_used = Channel_H + sqrt(cfg.csiErrVar)*E;
        end

        % ===== Tx Signal =====
        Symbol_X = make_signal(cfg, N_t, L);

        % ===== Statistical prior (mean/var) for VAMP family =====
        [pri_mean, pri_var] = choose_stat_prior(cfg, Symbol_X);

        % ===== I/O =====
        Chan_Symbol = Channel_H * Symbol_X;
        Rec_Y       = Chan_Symbol + noise;
        Rec_vec_y   = Rec_Y(:);

        % ===== Denoise Y (QD) =====
        % [Rec_Y_NL_vec, tau, U_P] = DenoisedSignal_QD_trained(Hamil_P_base, Rec_vec_y,beta_opt);
        [Rec_Y_NL_vec, tau, U_P] = DenoisedSignal_QD(Hamil_P_base, Rec_vec_y, Chan_Symbol);
 

        Rec_Y_NL = reshape(Rec_Y_NL_vec, N_r, L);

        % ===== Y NMSE =====
        eNL = norm(Rec_Y_NL_vec - Chan_Symbol(:))^2 / max(norm(Chan_Symbol(:))^2, eps);
        eY  = norm(Rec_vec_y    - Chan_Symbol(:))^2 / max(norm(Chan_Symbol(:))^2, eps);
        if ~isfinite(eNL), eNL = 0; end
        if ~isfinite(eY ), eY  = 0; end
        err_y_NL(iter) = err_y_NL(iter) + eNL;
        err_y(iter)    = err_y(iter)    + eY;

        % ===== LS baselines =====
        Hpinv  = pinv(H_used);
        LS_Ori = Hpinv * Rec_Y;
        LS_QD  = Hpinv * Rec_Y_NL;

        err_x_LS(iter)    = err_x_LS(iter)    + nmse(LS_Ori, Symbol_X);
        err_x_LS_QD(iter) = err_x_LS_QD(iter) + nmse(LS_QD,  Symbol_X);

        % ===== Robust baselines (NEW) =====
        % Solve per snapshot column: y(:,l) = H_used x(:,l)
        Xhat_LAD    = zeros(N_t, L);
        Xhat_Huber  = zeros(N_t, L);
        Xhat_LAD_QD = zeros(N_t, L);
        Xhat_Huber_QD = zeros(N_t, L);

        for l = 1:L
            ycol  = Rec_Y(:,l);
            ycolQ = Rec_Y_NL(:,l);

            Xhat_LAD(:,l)      = robust_lad_irls(H_used, ycol,  rob);
            Xhat_Huber(:,l)    = robust_huber_irls(H_used, ycol, rob);

            Xhat_LAD_QD(:,l)   = robust_lad_irls(H_used, ycolQ,  rob);
            Xhat_Huber_QD(:,l) = robust_huber_irls(H_used, ycolQ, rob);
        end

        err_x_LAD(iter)      = err_x_LAD(iter)      + nmse(Xhat_LAD, Symbol_X);
        err_x_Huber(iter)    = err_x_Huber(iter)    + nmse(Xhat_Huber, Symbol_X);
        err_x_LAD_QD(iter)   = err_x_LAD_QD(iter)   + nmse(Xhat_LAD_QD, Symbol_X);
        err_x_Huber_QD(iter) = err_x_Huber_QD(iter) + nmse(Xhat_Huber_QD, Symbol_X);

        % ===== Nonlinear LS on Y_tilde =====
        mat_A = U_P * diag(tau) * U_P' * kron(eye(L), H_used);
        Xhat_NL = QD_NL_LS(H_used, U_P, tau, Rec_Y_NL, Symbol_X);
        Xhat_NL = reshape(Xhat_NL, N_t, L);
        err_x_NL_LS_QD(iter) = err_x_NL_LS_QD(iter) + nmse(Xhat_NL, Symbol_X);

        % ===== VAMP variants =====
        [est_X_VAMP_ori,~] = Recovery_VAMP(Rec_vec_y, kron(eye(L),H_used), pri_mean, pri_var, var_noise);
        if all(isfinite(est_X_VAMP_ori(:)))
            err_x_VAMP_ori(iter) = err_x_VAMP_ori(iter) + nmse_vec(est_X_VAMP_ori, Symbol_X);
        end

        [est_X_VAMP_QD,~] = Recovery_VAMP(Rec_Y_NL_vec, mat_A, pri_mean, pri_var, var_noise);
        if all(isfinite(est_X_VAMP_QD(:)))
            err_x_VAMP_QD(iter) = err_x_VAMP_QD(iter) + nmse_vec(est_X_VAMP_QD, Symbol_X);
            frame_valid_QD = frame_valid_QD + 1;
        end

        est_X_PP = QD_VAMP_QAM_aware(Rec_Y_NL_vec, mat_A, pri_mean, pri_var, var_noise, Symbol_X(:), const_Hamil, Mconst);
        if all(isfinite(est_X_PP(:)))
            err_x_VAMP_PlugPlay(iter) = err_x_VAMP_PlugPlay(iter) + nmse_vec(est_X_PP, Symbol_X);
            frame_valid_PP = frame_valid_PP + 1;
        end

        % ===== BER =====
        if doBER_now
            bits_tx = qamdemod(Symbol_X, Mconst, 'UnitAveragePower', true, 'OutputType','bit');

            % LS on Y
            if all(isfinite(LS_Ori(:)))
                ber_LS(iter) = ber_LS(iter) + ber_from_est(LS_Ori, bits_tx, Mconst);
                ber_valid_LS = ber_valid_LS + 1;
            end

            % VAMP on Y
            if all(isfinite(est_X_VAMP_ori(:)))
                xv = est_X_VAMP_ori(:);
                if numel(xv) == N_t*L
                    Xv = reshape(xv, N_t, L);
                    ber_VAMP_ori(iter) = ber_VAMP_ori(iter) + ber_from_est(Xv, bits_tx, Mconst);
                    ber_valid_VAMP_ori = ber_valid_VAMP_ori + 1;
                end
            end

            % LS on Ytilde
            if all(isfinite(LS_QD(:)))
                ber_LS_QD(iter) = ber_LS_QD(iter) + ber_from_est(LS_QD, bits_tx, Mconst);
                ber_valid_LS_QD = ber_valid_LS_QD + 1;
            end

            % NL-LS on Ytilde
            if all(isfinite(Xhat_NL(:)))
                ber_NL_LS_QD(iter) = ber_NL_LS_QD(iter) + ber_from_est(Xhat_NL, bits_tx, Mconst);
                ber_valid_NL = ber_valid_NL + 1;
            end

            % QD-VAMP
            if all(isfinite(est_X_VAMP_QD(:)))
                xv = est_X_VAMP_QD(:);
                if numel(xv) == N_t*L
                    Xv = reshape(xv, N_t, L);
                    ber_VAMP_QD(iter) = ber_VAMP_QD(iter) + ber_from_est(Xv, bits_tx, Mconst);
                    ber_valid_VAMP_QD = ber_valid_VAMP_QD + 1;
                end
            end

            % PnP
            if all(isfinite(est_X_PP(:)))
                xpp = est_X_PP(:);
                if numel(xpp) == N_t*L
                    Xv = reshape(xpp, N_t, L);
                    ber_PnP(iter) = ber_PnP(iter) + ber_from_est(Xv, bits_tx, Mconst);
                    ber_valid_PnP = ber_valid_PnP + 1;
                end
            end

            % --- NEW: robust BER ---
            if all(isfinite(Xhat_LAD(:)))
                ber_LAD(iter) = ber_LAD(iter) + ber_from_est(Xhat_LAD, bits_tx, Mconst);
                ber_valid_LAD = ber_valid_LAD + 1;
            end
            if all(isfinite(Xhat_Huber(:)))
                ber_Huber(iter) = ber_Huber(iter) + ber_from_est(Xhat_Huber, bits_tx, Mconst);
                ber_valid_Huber = ber_valid_Huber + 1;
            end
            if all(isfinite(Xhat_LAD_QD(:)))
                ber_LAD_QD(iter) = ber_LAD_QD(iter) + ber_from_est(Xhat_LAD_QD, bits_tx, Mconst);
                ber_valid_LAD_QD = ber_valid_LAD_QD + 1;
            end
            if all(isfinite(Xhat_Huber_QD(:)))
                ber_Huber_QD(iter) = ber_Huber_QD(iter) + ber_from_est(Xhat_Huber_QD, bits_tx, Mconst);
                ber_valid_Huber_QD = ber_valid_Huber_QD + 1;
            end
        end
    end

    % ===== Average NMSE =====
    err_y_NL(iter) = err_y_NL(iter) / FrameAll;
    err_y(iter)    = err_y(iter)    / FrameAll;

    err_x_LS(iter)          = err_x_LS(iter)          / FrameAll;
    err_x_LS_QD(iter)       = err_x_LS_QD(iter)       / FrameAll;
    err_x_NL_LS_QD(iter)    = err_x_NL_LS_QD(iter)    / FrameAll;
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

    % ===== Average BER =====
    if doBER_now
        if ber_valid_LS > 0,        ber_LS(iter)       = ber_LS(iter)       / ber_valid_LS;        else, ber_LS(iter)=NaN; end
        if ber_valid_VAMP_ori > 0,  ber_VAMP_ori(iter) = ber_VAMP_ori(iter) / ber_valid_VAMP_ori;  else, ber_VAMP_ori(iter)=NaN; end
        if ber_valid_LS_QD > 0,     ber_LS_QD(iter)    = ber_LS_QD(iter)    / ber_valid_LS_QD;     else, ber_LS_QD(iter)=NaN; end
        if ber_valid_NL > 0,        ber_NL_LS_QD(iter) = ber_NL_LS_QD(iter) / ber_valid_NL;        else, ber_NL_LS_QD(iter)=NaN; end
        if ber_valid_VAMP_QD > 0,   ber_VAMP_QD(iter)  = ber_VAMP_QD(iter)  / ber_valid_VAMP_QD;   else, ber_VAMP_QD(iter)=NaN; end
        if ber_valid_PnP > 0,       ber_PnP(iter)      = ber_PnP(iter)      / ber_valid_PnP;       else, ber_PnP(iter)=NaN; end

        if ber_valid_LAD > 0,       ber_LAD(iter)      = ber_LAD(iter)      / ber_valid_LAD;       else, ber_LAD(iter)=NaN; end
        if ber_valid_Huber > 0,     ber_Huber(iter)    = ber_Huber(iter)    / ber_valid_Huber;     else, ber_Huber(iter)=NaN; end
        if ber_valid_LAD_QD > 0,    ber_LAD_QD(iter)   = ber_LAD_QD(iter)   / ber_valid_LAD_QD;    else, ber_LAD_QD(iter)=NaN; end
        if ber_valid_Huber_QD > 0,  ber_Huber_QD(iter) = ber_Huber_QD(iter) / ber_valid_Huber_QD;  else, ber_Huber_QD(iter)=NaN; end
    end
end

% ===== Titles =====
noiseTypeStr = sprintf('$\\alpha$-stable noise ($\\alpha=%g$)', cfg.alpha);
if cfg.useColored, noiseTypeStr = [noiseTypeStr, ', colored']; else, noiseTypeStr = [noiseTypeStr, ', white']; end

switch lower(cfg.sigModel)
    case 'qam', sigTypeStr = sprintf('%s symbols', cfg.mod);
    case 'gmm', sigTypeStr = 'GMM signal';
    case 'gaussian', sigTypeStr = 'Gaussian signal';
    case 'noprior', sigTypeStr = 'no-prior signal';
    otherwise, sigTypeStr = 'unknown signal';
end

chanStr = cfg.fadingModel;
if cfg.useImperfectCSI, chanStr = [chanStr, ', imperfect CSI']; else, chanStr = [chanStr, ', perfect CSI']; end
if cfg.useCorrChan, chanStr = [chanStr, ', correlated']; else, chanStr = [chanStr, ', i.i.d.']; end

priorStr = sprintf('priorCase=%s', cfg.priorCase);

%% ================= Figure: NMSE in X =================
figure
semilogy(SNR_Range, err_x_LS, '-o', 'LineWidth', 2); hold on
semilogy(SNR_Range, err_x_VAMP_ori, '-^', 'LineWidth', 2);
semilogy(SNR_Range, err_x_LS_QD, '-s', 'LineWidth', 2);
semilogy(SNR_Range, err_x_NL_LS_QD, '-d', 'LineWidth', 2);
semilogy(SNR_Range, err_x_VAMP_QD, '->', 'LineWidth', 2);
semilogy(SNR_Range, err_x_VAMP_PlugPlay, '-<', 'LineWidth', 2);

% NEW robust curves
semilogy(SNR_Range, err_x_LAD, '--o', 'LineWidth', 1.8);
semilogy(SNR_Range, err_x_Huber, '--^', 'LineWidth', 1.8);
semilogy(SNR_Range, err_x_LAD_QD, '--s', 'LineWidth', 1.8);
semilogy(SNR_Range, err_x_Huber_QD, '--d', 'LineWidth', 1.8);

grid on;
xlabel('SNR (dB)','Interpreter','latex');
ylabel('Estimation error in $\mathbf{X}$','Interpreter','latex');
legend({'LS on $\mathbf{Y}$','VAMP on $\mathbf{Y}$', ...
        'LS on $\tilde{\mathbf{Y}}$', 'NL-LS on $\tilde{\mathbf{Y}}$', ...
        'QD-VAMP on $\tilde{\mathbf{Y}}$', 'PnP-VAMP-QD on $\tilde{\mathbf{Y}}$', ...
        'LAD($\ell_1$) on $\mathbf{Y}$','Huber on $\mathbf{Y}$', ...
        'LAD($\ell_1$) on $\tilde{\mathbf{Y}}$','Huber on $\tilde{\mathbf{Y}}$'}, ...
        'Interpreter','latex','Location','best');
title(sprintf('%s, %s, %s, %s', noiseTypeStr, sigTypeStr, chanStr, priorStr),'Interpreter','latex');

%% ================= Figure: BER =================
if doBER_now
    figure;
    semilogy(SNR_Range, ber_LS,       '-o', 'LineWidth', 2); hold on;
    semilogy(SNR_Range, ber_VAMP_ori, '-^', 'LineWidth', 2);
    semilogy(SNR_Range, ber_LS_QD,    '-s', 'LineWidth', 2);
    semilogy(SNR_Range, ber_NL_LS_QD, '-d', 'LineWidth', 2);
    semilogy(SNR_Range, ber_VAMP_QD,  '->', 'LineWidth', 2);
    semilogy(SNR_Range, ber_PnP,      '-<', 'LineWidth', 2);

    % NEW robust BER
    semilogy(SNR_Range, ber_LAD,      '--o', 'LineWidth', 1.8);
    semilogy(SNR_Range, ber_Huber,    '--^', 'LineWidth', 1.8);
    semilogy(SNR_Range, ber_LAD_QD,   '--s', 'LineWidth', 1.8);
    semilogy(SNR_Range, ber_Huber_QD, '--d', 'LineWidth', 1.8);

    grid on;
    xlabel('SNR (dB)'); ylabel('BER');
    legend('LS on Y','VAMP on Y','LS on \tilde{Y}','NL-LS on \tilde{Y}', ...
           'QD-VAMP on \tilde{Y}','PnP-VAMP-QD on \tilde{Y}', ...
           'LAD on Y','Huber on Y','LAD on \tilde{Y}','Huber on \tilde{Y}', ...
           'Location','best');
    title(sprintf('BER: %s, %s, %s', sigTypeStr, chanStr, priorStr), 'Interpreter','none');
end


%% ================= Helpers =================
function [pri_mean, pri_var] = choose_stat_prior(cfg, Symbol_X)
    true_mean = mean(Symbol_X(:));
    true_var  = var(Symbol_X(:), 1);
    true_var = max(true_var, 1e-12);

    switch lower(cfg.priorCase)
        case 'known'
            pri_mean = true_mean;
            pri_var  = true_var;

        case 'partial'
            pm = true_mean; pv = true_var;
            if ~isfield(cfg,'partialMode'), cfg.partialMode='var_mismatch'; end
            switch lower(cfg.partialMode)
                case 'var_mismatch'
                    f = cfg.varMismatchFactor;
                    pri_mean = pm;
                    pri_var  = max(f * pv, 1e-12);
                case 'mean_mismatch'
                    b = cfg.meanMismatchBias;
                    pri_mean = pm + b;
                    pri_var  = pv;
                case 'both'
                    f = cfg.varMismatchFactor; b = cfg.meanMismatchBias;
                    pri_mean = pm + b;
                    pri_var  = max(f * pv, 1e-12);
                otherwise
                    error('Unknown cfg.partialMode');
            end

        case 'absent'
            if ~isfield(cfg,'absentMode'), cfg.absentMode='default01'; end
            switch lower(cfg.absentMode)
                case 'default01'
                    pri_mean = 0; pri_var = 1;
                case 'weak'
                    pri_mean = 0; pri_var = cfg.weakPriorVar;
                otherwise
                    error('Unknown cfg.absentMode');
            end

        otherwise
            error('Unknown cfg.priorCase');
    end
end



function ber = ber_from_est(Xhat, bits_tx, M)
    idx  = qamdemod(Xhat, M, 'UnitAveragePower', true);
    Xdec = qammod(idx,  M, 'UnitAveragePower', true);
    bits = qamdemod(Xdec, M, 'UnitAveragePower', true, 'OutputType','bit');
    ber  = mean(bits(:) ~= bits_tx(:));
end







function v = nmse(A,B)
    v = norm(A - B, 'fro')^2 / max(norm(B, 'fro')^2, eps);
end

function v = nmse_vec(x_vec, B)
    v = norm(x_vec - B(:))^2 / max(norm(B(:))^2, eps);
end



function [x_est, nmse_val] = QD_NL_LS(H, Psi, tau, Y_denoised, X_true)
    max_iter = 800;
    eta = 1e-3;
    [~, L] = size(Y_denoised);
    N_t = size(H, 2);

    y_vec = Y_denoised(:);
    x0 = pinv(H) * Y_denoised;
    x0 = x0(:);

    PsiT = Psi';
    Tau  = diag(tau);
    H_big = kron(eye(L), H);

    A = Psi * Tau * PsiT * H_big;
    f_x = @(x) A * x;

    x = x0;
    for iter = 1:max_iter
        res  = y_vec - f_x(x);
        grad = -2 * (A') * res;
        x = x - eta * grad;
    end

    x_est = reshape(x, N_t, L);
    nmse_val = norm(x_est - X_true, 'fro')^2 / max(norm(X_true, 'fro')^2, eps);
end

%% ===== Robust baseline: LAD via IRLS =====
function x = robust_lad_irls(H, y, rob)
    % min_x ||y - Hx||_1 (complex-safe via |r|)
    [~, n] = size(H);
    % init with LS
    x = pinv(H) * y;

    for it = 1:rob.maxIter
        r = y - H*x;
        w = 1 ./ max(abs(r), rob.epsW);   % LAD weights
        % Weighted LS: (H'*W*H)x = H'*W*y
        % Using implicit diag(w) without forming big matrix:
        Hw = bsxfun(@times, H, w);        % each row scaled by w 
        x_new = (Hw' * H) \ (Hw' * y);    % equivalent to H'W H and H'W y

        if norm(x_new - x)^2 <= rob.tol * max(norm(x)^2, 1)
            x = x_new; return;
        end
        x = x_new;
    end
end

%% ===== Robust baseline: Huber via IRLS =====
function x = robust_huber_irls(H, y, rob)
    % Huber loss: quadratic near 0, linear in tails
    [~, n] = size(H);
    x = pinv(H) * y;

    for it = 1:rob.maxIter
        r = y - H*x;
        % robust scale via MAD of |r|
        s = mad(abs(r), 1) / 1.1774;
        s = max(s, 1e-12);
        c = rob.huber_c * s;

        ar = abs(r);
        w = ones(size(r));
        idx = ar > c;
        w(idx) = c ./ max(ar(idx), rob.epsW);   % Huber weights

        Hw = bsxfun(@times, H, w);
        x_new = (Hw' * H) \ (Hw' * y);

        if norm(x_new - x)^2 <= rob.tol * max(norm(x)^2, 1)
            x = x_new; return;
        end
        x = x_new;
    end
end