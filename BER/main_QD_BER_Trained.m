%% ================= BER =================

clear; clc; rng(1);

% ---------- Dimensions ----------
N_r = 16; N_t = 8; L = 1;
SNR_Range = -5:5:20;
FrameAll = 1000;

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
 

% Signal model:
cfg.mod   = '64qam';     
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

%% Training parameter beta first
% MC_train=1;
% beta_SNR = ParaTrain(SNR_Range,MC_train,cfg,N_r, N_t, L, var_channel,P_sig, delta,Hamil_P_base);
beta_SNR = [0.01,-70,-0.1,-74,-13,-13; 0.01,0.03,-300,0.3,1.4,0.8];

for iter = 1:numel(SNR_Range)
    SNR_dB    = SNR_Range(iter);
    var_noise = P_sig / 10^(SNR_dB/10);
    beta_opt_all = beta_SNR(:, iter); 

    frame_valid_QD = 0;
    frame_valid_PP = 0;

    % BER valid counters
    ber_valid_LS = 0; ber_valid_VAMP_ori = 0; ber_valid_LS_QD = 0;
    ber_valid_NL = 0; ber_valid_VAMP_QD = 0;  ber_valid_PnP = 0;
    ber_valid_LAD = 0; ber_valid_Huber = 0; ber_valid_LAD_QD = 0; ber_valid_Huber_QD = 0;

    for frame = 1:FrameAll
        % ===== Noise =====
        noise = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta);
        gamma = (var_noise/2)^(1/cfg.alpha);
        % ===== Channel =====
        [Channel_H, stats] = make_channel(cfg, N_r, N_t, var_channel);
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
        Symbol_X = qammod(randi([0 cfg.Mconst-1],N_t,L),cfg.Mconst,'UnitAveragePower',true);
        % ===== Statistical prior (mean/var) for VAMP family ===== 
        [pri_mean, pri_var] = choose_stat_prior_X(cfg, Symbol_X);

        % ===== I/O =====
        Chan_Symbol = Channel_H * Symbol_X;
        Rec_Y       = Chan_Symbol + noise;
        Rec_vec_y   = Rec_Y(:);

        % ===== Denoise Y (QD) ===== 
%         beta_opt_all = [0,-1000];
        [Rec_Y_NL_vec, tau, U_P] = DenoisedSignal_QD_trained(Hamil_P_base, Rec_vec_y,beta_opt_all);
        % [Rec_Y_NL_vec, tau, U_P] = DenoisedSignal_QD(Hamil_P_base, Rec_vec_y, Chan_Symbol);

        Rec_Y_NL = reshape(Rec_Y_NL_vec, N_r, L);
 

        % ===== LS baselines =====
        Hpinv  = pinv(H_used);
        LS_Ori = Hpinv * Rec_Y;
        LS_QD  = Hpinv * Rec_Y_NL;
 

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

 

        % ===== Nonlinear LS on Y_tilde =====
        mat_A = U_P * diag(tau) * U_P' * kron(eye(L), H_used);
 

        % ===== VAMP variants =====
        [est_X_VAMP_ori,~] = Recovery_VAMP(Rec_vec_y, kron(eye(L),H_used), pri_mean, pri_var, gamma);
        % QD-VAMP
        [est_X_VAMP_QD,~] = Recovery_VAMP(Rec_Y_NL_vec, mat_A, pri_mean, pri_var, gamma);
        % PnP-VAMP-QD
        beta_opt = [-73,-0.05];
        est_X_PP = QD_VAMP_QAM_aware_trained(Rec_Y_NL_vec, mat_A, pri_mean, pri_var, gamma, const_Hamil, cfg.Mconst,beta_opt);
        % est_X_PP = QD_VAMP_QAM_aware(Rec_Y_NL_vec, mat_A, pri_mean, pri_var, gamma,Symbol_X(:),const_Hamil, cfg.Mconst);

        % ===== BER =====
            bits_tx = qamdemod(Symbol_X, cfg.Mconst, 'UnitAveragePower', true, 'OutputType','bit');

            % LS on Y
            if all(isfinite(LS_Ori(:)))
                ber_LS(iter) = ber_LS(iter) + ber_from_est(LS_Ori, bits_tx, cfg.Mconst);
                ber_valid_LS = ber_valid_LS + 1;
            end

            % VAMP on Y
            if all(isfinite(est_X_VAMP_ori(:)))
                xv = est_X_VAMP_ori(:);
                if numel(xv) == N_t*L
                    Xv = reshape(xv, N_t, L);
                    ber_VAMP_ori(iter) = ber_VAMP_ori(iter) + ber_from_est(Xv, bits_tx, cfg.Mconst);
                    ber_valid_VAMP_ori = ber_valid_VAMP_ori + 1;
                end
            end

            % LS on Ytilde
            if all(isfinite(LS_QD(:)))
                ber_LS_QD(iter) = ber_LS_QD(iter) + ber_from_est(LS_QD, bits_tx, cfg.Mconst);
                ber_valid_LS_QD = ber_valid_LS_QD + 1;
            end


            % QD-VAMP
            if all(isfinite(est_X_VAMP_QD(:)))
                xv = est_X_VAMP_QD(:);
                if numel(xv) == N_t*L
                    Xv = reshape(xv, N_t, L);
                    ber_VAMP_QD(iter) = ber_VAMP_QD(iter) + ber_from_est(Xv, bits_tx, cfg.Mconst);
                    ber_valid_VAMP_QD = ber_valid_VAMP_QD + 1;
                end
            end

            % PnP
            if all(isfinite(est_X_PP(:)))
                xpp = est_X_PP(:);
                if numel(xpp) == N_t*L
                    Xv = reshape(xpp, N_t, L);
                    ber_PnP(iter) = ber_PnP(iter) + ber_from_est(Xv, bits_tx, cfg.Mconst);
                    ber_valid_PnP = ber_valid_PnP + 1;
                end
            end

            % --- NEW: robust BER ---
            if all(isfinite(Xhat_LAD(:)))
                ber_LAD(iter) = ber_LAD(iter) + ber_from_est(Xhat_LAD, bits_tx, cfg.Mconst);
                ber_valid_LAD = ber_valid_LAD + 1;
            end
            if all(isfinite(Xhat_Huber(:)))
                ber_Huber(iter) = ber_Huber(iter) + ber_from_est(Xhat_Huber, bits_tx, cfg.Mconst);
                ber_valid_Huber = ber_valid_Huber + 1;
            end
            if all(isfinite(Xhat_LAD_QD(:)))
                ber_LAD_QD(iter) = ber_LAD_QD(iter) + ber_from_est(Xhat_LAD_QD, bits_tx, cfg.Mconst);
                ber_valid_LAD_QD = ber_valid_LAD_QD + 1;
            end
            if all(isfinite(Xhat_Huber_QD(:)))
                ber_Huber_QD(iter) = ber_Huber_QD(iter) + ber_from_est(Xhat_Huber_QD, bits_tx, cfg.Mconst);
                ber_valid_Huber_QD = ber_valid_Huber_QD + 1;
            end
    end

    % ===== Average BER =====
        if ber_valid_LS > 0,        ber_LS(iter)       = ber_LS(iter)       / ber_valid_LS;        else, ber_LS(iter)=NaN; end
        if ber_valid_VAMP_ori > 0,  ber_VAMP_ori(iter) = ber_VAMP_ori(iter) / ber_valid_VAMP_ori;  else, ber_VAMP_ori(iter)=NaN; end
        if ber_valid_LS_QD > 0,     ber_LS_QD(iter)    = ber_LS_QD(iter)    / ber_valid_LS_QD;     else, ber_LS_QD(iter)=NaN; end
        if ber_valid_VAMP_QD > 0,   ber_VAMP_QD(iter)  = ber_VAMP_QD(iter)  / ber_valid_VAMP_QD;   else, ber_VAMP_QD(iter)=NaN; end
        if ber_valid_PnP > 0,       ber_PnP(iter)      = ber_PnP(iter)      / ber_valid_PnP;       else, ber_PnP(iter)=NaN; end

        if ber_valid_LAD > 0,       ber_LAD(iter)      = ber_LAD(iter)      / ber_valid_LAD;       else, ber_LAD(iter)=NaN; end
        if ber_valid_Huber > 0,     ber_Huber(iter)    = ber_Huber(iter)    / ber_valid_Huber;     else, ber_Huber(iter)=NaN; end
        if ber_valid_LAD_QD > 0,    ber_LAD_QD(iter)   = ber_LAD_QD(iter)   / ber_valid_LAD_QD;    else, ber_LAD_QD(iter)=NaN; end
        if ber_valid_Huber_QD > 0,  ber_Huber_QD(iter) = ber_Huber_QD(iter) / ber_valid_Huber_QD;  else, ber_Huber_QD(iter)=NaN; end
end

% ===== Titles =====
noiseTypeStr = sprintf('Trained, $\\alpha=%g$', cfg.alpha);

sigTypeStr = sprintf('%s', cfg.mod);


chanStr = cfg.fadingModel;
if cfg.useImperfectCSI, chanStr = [chanStr, ', imperfect CSI']; else, chanStr = [chanStr, ', perfect CSI']; end
if cfg.useCorrChan, chanStr = [chanStr, ', correlated']; else, chanStr = [chanStr, ', i.i.d.']; end

priorStr = sprintf('prior=%s', cfg.priorCase);


%% ================= Figure: BER =================
    figure;
    semilogy(SNR_Range, ber_LS,       '-o', 'LineWidth', 2); hold on;
    semilogy(SNR_Range, ber_VAMP_ori, '-^', 'LineWidth', 2);
    semilogy(SNR_Range, ber_LS_QD,    '-s', 'LineWidth', 2);
    semilogy(SNR_Range, ber_VAMP_QD,  '->', 'LineWidth', 2);
    semilogy(SNR_Range, ber_PnP,      '-<', 'LineWidth', 2);

    % NEW robust BER
    semilogy(SNR_Range, ber_LAD,      '--o', 'LineWidth', 1.8);
    semilogy(SNR_Range, ber_Huber,    '--^', 'LineWidth', 1.8);
    semilogy(SNR_Range, ber_LAD_QD,   '--s', 'LineWidth', 1.8);
    semilogy(SNR_Range, ber_Huber_QD, '--d', 'LineWidth', 1.8);

    grid on;
    xlabel('SNR (dB)'); ylabel('BER');
    legend('LS on Y','VAMP on Y','LS on \tilde{Y}', ...
           'QD-VAMP on \tilde{Y}','PnP-VAMP-QD on \tilde{Y}', ...
           'LAD on Y','Huber on Y','LAD on \tilde{Y}','Huber on \tilde{Y}', ...
           'Location','best');
    title(sprintf('%s,%s, %s, %s', noiseTypeStr, chanStr, priorStr), 'Interpreter','latex');
 
 
function ber = ber_from_est(Xhat, bits_tx, M)
    bits = qamdemod(Xhat, M, 'UnitAveragePower', true, 'OutputType','bit');
    ber  = mean(bits(:) ~= bits_tx(:));
end

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