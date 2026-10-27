%% ================= Experiment: α-stable / Colored noise / Complex baseband / Fading / (Imperfect CSI) / NMSE + BER =================
% NOTE:
% - BER uses qammod/qamdemod (Communications Toolbox).
% - To avoid "M undefined", we precompute M once per SNR loop.
% - To avoid reshape crashes (PnP/VAMP output length mismatch), we vectorize and check numel()==N_t*L.

clear; clc; rng(1);

% ---------- Dimensions ----------
N_r = 16; N_t = 8; L = 1;
SNR_Range = -5:5:20;
FrameAll = 100;

% ---------- Channel power ----------
var_channel = 1;

% ---------- Config (one place to change) ----------
cfg.alpha      = 1;           % 0<alpha<=2; alpha=2 -> Gaussian
cfg.useColored = false;       % color the noise?
cfg.h          = [1 -1];      % FIR for colored noise (e.g., high-pass)
cfg.colorMode  = 'spatial';   % 'spatial' or 'temporal' (optional; used if useColored=true)

% Signal model:
% 'qam'
cfg.sigModel   = 'qam';       % <-- for BER experiments, use 'qam'
cfg.pi_mix     = [0.5 0.5];   % GMM mixing weights (sum=1)
cfg.sigma_mix  = [1 10];      % GMM stds, zero-mean components

% ---------- Wireless realism toggles ----------
cfg.isComplex       = true;         % complex baseband?
cfg.fadingModel     = 'rayleigh';   % 'rayleigh' | 'rician'
cfg.K_rician        = 5;            % Rician K-factor (linear), used if rician
cfg.useCorrChan     = true;        % correlated MIMO?
cfg.rho_r           = 0.7;          % Rx correlation
cfg.rho_t           = 0.5;          % Tx correlation

cfg.useImperfectCSI = false;        % use H_hat for recovery?
cfg.csiErrVar       = 1e-2;         % variance of CSI error per entry

% ---------- Detection / BER ----------
cfg.mod   = '16QAM';         % 'QPSK' | '16QAM' | '64QAM'
cfg.doBER = true;            % compute BER (requires qammod/qamdemod)

% ---------- PRIOR CASE (statistical prior: mean/variance) ----------
% 'known' | 'partial' | 'absent'
cfg.priorCase = 'known';

% for partial-prior: mismatch settings (you can sweep these)
cfg.partialMode = 'var_mismatch';   % 'var_mismatch' | 'mean_mismatch' | 'both'
cfg.varMismatchFactor = 2.0;        % pri_var = factor * true_var
cfg.meanMismatchBias  = 0.3;        % pri_mean = true_mean + bias (for mean mismatch)

% for absent-prior: choose default or weak prior
cfg.absentMode = 'default01';       % 'default01' | 'weak'
cfg.weakPriorVar = 1e3;             % used if absentMode='weak'

% ---------- For α-stable noise ----------
P_sig = 1;     % target signal power for SNR definition (used to set var_noise baseline)
delta = 0;     % location (mean) for noise

% ---------- quantum/Hamiltonian items ----------
hbar = 1; mass = 1;
const_Hamil = hbar^2 / (2*mass);
Hamil_P_base = Hamiltonian_Original(N_r, L, const_Hamil);

% ---------- Collectors: NMSE ----------
err_y_NL = zeros(1, numel(SNR_Range));
err_y    = zeros(1, numel(SNR_Range));

err_x_LS_QD          = zeros(1, numel(SNR_Range));
err_x_LS             = zeros(1, numel(SNR_Range));
err_x_NL_LS_QD       = zeros(1, numel(SNR_Range));
err_x_VAMP_QD        = zeros(1, numel(SNR_Range));
err_x_VAMP_PlugPlay  = zeros(1, numel(SNR_Range));
err_x_VAMP_ori       = zeros(1, numel(SNR_Range));

% ---------- Collectors: BER ----------
ber_LS       = zeros(1, numel(SNR_Range));
ber_VAMP_ori = zeros(1, numel(SNR_Range));
ber_LS_QD    = zeros(1, numel(SNR_Range));
ber_NL_LS_QD = zeros(1, numel(SNR_Range));
ber_VAMP_QD  = zeros(1, numel(SNR_Range));   % QD-VAMP
ber_PnP      = zeros(1, numel(SNR_Range));   % PnP-VAMP-QD

for iter = 1:numel(SNR_Range)
    SNR_dB   = SNR_Range(iter);
    var_noise = P_sig / 10^(SNR_dB/10);   % target noise var baseline (Gaussian case)

    frame_valid_QD = 0;
    frame_valid_PP = 0;

    % ---------- Precompute modulation order Mconst ----------
    doBER_now = isfield(cfg,'doBER') && cfg.doBER && strcmpi(cfg.sigModel,'qam');
    if doBER_now
        Mconst = get_mod_order(cfg.mod);
    else
        Mconst = [];
    end

    % BER valid counters per SNR
    ber_valid_LS = 0; ber_valid_VAMP_ori = 0; ber_valid_LS_QD = 0;
    ber_valid_NL = 0; ber_valid_VAMP_QD = 0;  ber_valid_PnP = 0;

    for frame = 1:FrameAll
        % ===== Noise: α-stable (colored optional) =====
        noise = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta);

        % ===== Channel =====
        Channel_H = make_channel(cfg, N_r, N_t, var_channel);

        % ===== (Optional) Imperfect CSI used in recovery only =====
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

        % ===== Choose statistical prior (pri_mean, pri_var) for this frame =====
        [pri_mean, pri_var] = choose_stat_prior(cfg, Symbol_X);

        % ===== I/O =====
        Chan_Symbol = Channel_H * Symbol_X;   % true channel for data generation
        Rec_Y       = Chan_Symbol + noise;
        Rec_vec_y   = Rec_Y(:);

        % ===== Denoise Y =====
        [Rec_Y_NL_vec, tau, U_P] = DenoisedSignal_QD(Hamil_P_base, Rec_vec_y, Chan_Symbol);
        Rec_Y_NL = reshape(Rec_Y_NL_vec, N_r, L);

        % ===== Y NMSE  =====
        eNL = norm(Rec_Y_NL_vec - Chan_Symbol(:))^2 / max(norm(Chan_Symbol(:))^2, eps);
        eY  = norm(Rec_vec_y    - Chan_Symbol(:))^2 / max(norm(Chan_Symbol(:))^2, eps);
        if ~isfinite(eNL), eNL = 0; end
        if ~isfinite(eY ), eY  = 0; end
        err_y_NL(iter) = err_y_NL(iter) + eNL;
        err_y(iter)    = err_y(iter)    + eY;

        % ===== LS on Y / Y_tilde (use H_used for recovery) =====
        Hpinv  = pinv(H_used);
        LS_Ori = Hpinv * Rec_Y;
        LS_QD  = Hpinv * Rec_Y_NL;

        err_x_LS(iter)      = err_x_LS(iter)      + nmse(LS_Ori, Symbol_X);
        err_x_LS_QD(iter)   = err_x_LS_QD(iter)   + nmse(LS_QD,  Symbol_X);

        % ===== Nonlinear LS on Y_tilde =====
        mat_A = U_P * diag(tau) * U_P' * kron(eye(L), H_used);
        Xhat_NL = QD_NL_LS(H_used, U_P, tau, Rec_Y_NL, Symbol_X);
        Xhat_NL = reshape(Xhat_NL, N_t, L);
        err_x_NL_LS_QD(iter) = err_x_NL_LS_QD(iter) + nmse(Xhat_NL, Symbol_X);

        % ===== VAMP variants (external) =====
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

        % ===== BER (extended to all estimators; safe reshape + M always defined) =====
        if doBER_now
            bits_tx = qamdemod(Symbol_X, Mconst, 'UnitAveragePower', true, 'OutputType','bit');

            % LS on Y
            if all(isfinite(LS_Ori(:)))
                ber_LS(iter) = ber_LS(iter) + ber_from_est(LS_Ori, bits_tx, Mconst);
                ber_valid_LS = ber_valid_LS + 1;
            end

            % VAMP on Y (original)
            if all(isfinite(est_X_VAMP_ori(:)))
                xv = est_X_VAMP_ori(:);
                if numel(xv) == N_t*L
                    Xv = reshape(xv, N_t, L);
                    ber_VAMP_ori(iter) = ber_VAMP_ori(iter) + ber_from_est(Xv, bits_tx, Mconst);
                    ber_valid_VAMP_ori = ber_valid_VAMP_ori + 1;
                end
            end

            % LS on Y_tilde
            if all(isfinite(LS_QD(:)))
                ber_LS_QD(iter) = ber_LS_QD(iter) + ber_from_est(LS_QD, bits_tx, Mconst);
                ber_valid_LS_QD = ber_valid_LS_QD + 1;
            end

            % NL-LS on Y_tilde
            if all(isfinite(Xhat_NL(:)))
                ber_NL_LS_QD(iter) = ber_NL_LS_QD(iter) + ber_from_est(Xhat_NL, bits_tx, Mconst);
                ber_valid_NL = ber_valid_NL + 1;
            end

            % QD-VAMP on Y_tilde
            if all(isfinite(est_X_VAMP_QD(:)))
                xv = est_X_VAMP_QD(:);
                if numel(xv) == N_t*L
                    Xv = reshape(xv, N_t, L);
                    ber_VAMP_QD(iter) = ber_VAMP_QD(iter) + ber_from_est(Xv, bits_tx, Mconst);
                    ber_valid_VAMP_QD = ber_valid_VAMP_QD + 1;
                end
            end

            % PnP-VAMP on Y_tilde (safe length check)
            if all(isfinite(est_X_PP(:)))
                xpp = est_X_PP(:);
                if numel(xpp) == N_t*L
                    Xv = reshape(xpp, N_t, L);
                    ber_PnP(iter) = ber_PnP(iter) + ber_from_est(Xv, bits_tx, Mconst);
                    ber_valid_PnP = ber_valid_PnP + 1;
                end
            end
        end
    end

    % ===== Average over frames: NMSE =====
    err_y_NL(iter) = err_y_NL(iter) / FrameAll;
    err_y(iter)    = err_y(iter)    / FrameAll;

    err_x_LS(iter)        = err_x_LS(iter)        / FrameAll;
    err_x_LS_QD(iter)     = err_x_LS_QD(iter)     / FrameAll;
    err_x_NL_LS_QD(iter)  = err_x_NL_LS_QD(iter)  / FrameAll;
    err_x_VAMP_ori(iter)  = err_x_VAMP_ori(iter)  / FrameAll;

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

    % ===== Average over frames: BER =====
    if doBER_now
        if ber_valid_LS > 0,        ber_LS(iter)       = ber_LS(iter)       / ber_valid_LS;        else, ber_LS(iter)=NaN; end
        if ber_valid_VAMP_ori > 0,  ber_VAMP_ori(iter) = ber_VAMP_ori(iter) / ber_valid_VAMP_ori;  else, ber_VAMP_ori(iter)=NaN; end
        if ber_valid_LS_QD > 0,     ber_LS_QD(iter)    = ber_LS_QD(iter)    / ber_valid_LS_QD;     else, ber_LS_QD(iter)=NaN; end
        if ber_valid_NL > 0,        ber_NL_LS_QD(iter) = ber_NL_LS_QD(iter) / ber_valid_NL;        else, ber_NL_LS_QD(iter)=NaN; end
        if ber_valid_VAMP_QD > 0,   ber_VAMP_QD(iter)  = ber_VAMP_QD(iter)  / ber_valid_VAMP_QD;   else, ber_VAMP_QD(iter)=NaN; end
        if ber_valid_PnP > 0,       ber_PnP(iter)      = ber_PnP(iter)      / ber_valid_PnP;       else, ber_PnP(iter)=NaN; end
    end
end

% ===== Auto title strings =====
noiseTypeStr = sprintf('$\\alpha$-stable noise ($\\alpha=%g$)', cfg.alpha);
if cfg.useColored
    noiseTypeStr = [noiseTypeStr, ', colored'];
else
    noiseTypeStr = [noiseTypeStr, ', white'];
end

switch lower(cfg.sigModel)
    case 'gmm'
        piStr = sprintf('[%.2g, %.2g]', cfg.pi_mix(1), cfg.pi_mix(2));
        sigTypeStr = sprintf('GMM signal (%s)', piStr);
    case 'gaussian'
        sigTypeStr = 'Gaussian signal';
    case 'noprior'
        sigTypeStr = 'no-prior signal';
    case 'qam'
        sigTypeStr = sprintf('%s symbols', cfg.mod);
    otherwise
        sigTypeStr = 'unknown signal';
end

chanStr = cfg.fadingModel;
if isfield(cfg,'useImperfectCSI') && cfg.useImperfectCSI
    chanStr = [chanStr, ', imperfect CSI'];
else
    chanStr = [chanStr, ', perfect CSI'];
end
if isfield(cfg,'useCorrChan') && cfg.useCorrChan
    chanStr = [chanStr, ', correlated'];
else
    chanStr = [chanStr, ', i.i.d.'];
end

priorStr = sprintf('priorCase=%s', cfg.priorCase);

%% ================= Figure: NMSE in X =================
figure
semilogy(SNR_Range, err_x_LS, '-o', 'LineWidth', 2); hold on
semilogy(SNR_Range, err_x_VAMP_ori, '-^', 'LineWidth', 2);
semilogy(SNR_Range, err_x_LS_QD, '-s', 'LineWidth', 2);
semilogy(SNR_Range, err_x_NL_LS_QD, '-d', 'LineWidth', 2);
semilogy(SNR_Range, err_x_VAMP_QD, '->', 'LineWidth', 2);
semilogy(SNR_Range, err_x_VAMP_PlugPlay, '-<', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)','Interpreter','latex');
ylabel('Estimation error in $\mathbf{X}$','Interpreter','latex');
legend({'LS on $\mathbf{Y}$','Original VAMP on $\mathbf{Y}$', ...
        'LS on $\tilde{\mathbf{Y}}$', 'Nonlinear LS on $\tilde{\mathbf{Y}}$', ...
        'VAMP on $\tilde{\mathbf{Y}}$ (QD-VAMP)', 'Plug-and-Play VAMP on $\tilde{\mathbf{Y}}$ (PnP)'}, ...
        'Interpreter','latex','Location','best');
title(sprintf('%s, %s, %s, %s', noiseTypeStr, sigTypeStr, chanStr, priorStr),'Interpreter','latex');

%% ================= Figure: BER (all estimators) =================
if isfield(cfg,'doBER') && cfg.doBER && strcmpi(cfg.sigModel,'qam')
    figure;
    semilogy(SNR_Range, ber_LS,       '-o', 'LineWidth', 2); hold on;
    semilogy(SNR_Range, ber_VAMP_ori, '-^', 'LineWidth', 2);
    semilogy(SNR_Range, ber_LS_QD,    '-s', 'LineWidth', 2);
    semilogy(SNR_Range, ber_NL_LS_QD, '-d', 'LineWidth', 2);
    semilogy(SNR_Range, ber_VAMP_QD,  '->', 'LineWidth', 2);
    semilogy(SNR_Range, ber_PnP,      '-<', 'LineWidth', 2);
    grid on;
    xlabel('SNR (dB)');
    ylabel('BER');
    legend('LS on Y','VAMP on Y','LS on \tilde{Y}','NL-LS on \tilde{Y}', ...
           'QD-VAMP on \tilde{Y}','PnP-VAMP on \tilde{Y}', ...
           'Location','best');
    title(sprintf('BER: %s, %s, %s', sigTypeStr, chanStr, priorStr), 'Interpreter','none');
end

%% ================= Helpers =================
function [pri_mean, pri_var] = choose_stat_prior(cfg, Symbol_X)
    % Statistical prior: only mean/variance (NOT constellation prior)
    true_mean = mean(Symbol_X(:));
    true_var  = var(Symbol_X(:), 1);             % population variance

    % Avoid degenerate prior var
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
                    f = 2.0;
                    if isfield(cfg,'varMismatchFactor'), f = cfg.varMismatchFactor; end
                    pri_mean = pm;
                    pri_var  = max(f * pv, 1e-12);

                case 'mean_mismatch'
                    b = 0.3;
                    if isfield(cfg,'meanMismatchBias'), b = cfg.meanMismatchBias; end
                    pri_mean = pm + b;
                    pri_var  = pv;

                case 'both'
                    f = 2.0; b = 0.3;
                    if isfield(cfg,'varMismatchFactor'), f = cfg.varMismatchFactor; end
                    if isfield(cfg,'meanMismatchBias'),  b = cfg.meanMismatchBias;  end
                    pri_mean = pm + b;
                    pri_var  = max(f * pv, 1e-12);

                otherwise
                    error('Unknown cfg.partialMode');
            end

        case 'absent'
            if ~isfield(cfg,'absentMode'), cfg.absentMode='default01'; end
            switch lower(cfg.absentMode)
                case 'default01'
                    pri_mean = 0;
                    pri_var  = 1;
                case 'weak'
                    pri_mean = 0;
                    pri_var  = cfg.weakPriorVar;
                otherwise
                    error('Unknown cfg.absentMode');
            end

        otherwise
            error('Unknown cfg.priorCase');
    end
end

function M = get_mod_order(modStr)
    switch upper(modStr)
        case 'QPSK',  M = 4;
        case '16QAM', M = 16;
        case '64QAM', M = 64;
        otherwise,    M = 16;
    end
end

function ber = ber_from_est(Xhat, bits_tx, M)
    % Hard decision to nearest QAM point then compute bit error rate
    idx  = qamdemod(Xhat, M, 'UnitAveragePower', true);
    Xdec = qammod(idx,  M, 'UnitAveragePower', true);
    bits = qamdemod(Xdec, M, 'UnitAveragePower', true, 'OutputType','bit');
    ber  = mean(bits(:) ~= bits_tx(:));
end

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
            X =  3 + 10*randn(N_t, L) ;

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
            C = V * diag(sqrt(d)) * V';   % Hermitian transpose

            N = C * N;
            N = N * (sqrt(var_noise) / max(std(N(:)), eps));
        else
            for r = 1:N_r
                N(r,:) = filter(cfg.h, 1, N(r,:));
            end
            if abs(cfg.alpha - 2) < 1e-12
                for r = 1:N_r
                    vr = var(N(r,:));
                    if vr > 0
                        N(r,:) = N(r,:) * sqrt(var_noise / vr);
                    end
                end
            else
                N = N * (sqrt(var_noise) / max(std(N(:)), eps));
            end
        end
    end
end

function v = nmse(A,B)
    v = norm(A - B, 'fro')^2 / max(norm(B, 'fro')^2, eps);
end

function v = nmse_vec(x_vec, B)
    v = norm(x_vec - B(:))^2 / max(norm(B(:))^2, eps);
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

function [x_est, nmse_val] = QD_NL_LS(H, Psi, tau, Y_denoised, X_true)
% QD_NL_LS  LS estimation under a FIXED QD operator A
% (Replaces gradient descent with closed-form regularized LS)

    [~, L] = size(Y_denoised);
    N_t = size(H, 2);

    % Vectorize observation
    y_vec = Y_denoised(:);

    % Build fixed linear operator A
    PsiT  = Psi';
    Tau   = diag(tau);
    H_big = kron(eye(L), H);
    A     = Psi * Tau * PsiT * H_big;

    % ---- Closed-form (regularized) LS ----
    reg = 1e-8;  % can tune: 1e-10 ~ 1e-6
    x = (A' * A + reg * eye(size(A,2))) \ (A' * y_vec);

    % Output
    x_est = reshape(x, N_t, L);
    nmse_val = norm(x_est - X_true, 'fro')^2 / max(norm(X_true, 'fro')^2, eps);
end