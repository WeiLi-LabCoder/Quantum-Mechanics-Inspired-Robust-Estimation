%% ================= Experiment: α-stable / Colored noise / Signal model =================
clear; clc;  rng(1);

% ---------- Dimensions ----------
N_r = 16; N_t = 8; L = 1;
SNR_Range = -5:5:20;
FrameAll = 3000;

% ---------- Channel power ----------
var_channel = 1;

% ---------- Config (one place to change) ----------
cfg.alpha      = 1;         % 0<alpha<=2; alpha=2 -> Gaussian
cfg.useColored = false;     % color the noise?
cfg.h          = [1 -1];    % FIR for colored noise (e.g., high-pass)

cfg.sigModel   = 'gaussian';      % 'gmm' | 'gaussian' | 'noprior'
cfg.pi_mix     = [0.5 0.5];       % GMM mixing weights (sum=1)
cfg.sigma_mix  = [1 10];          % GMM stds, zero-mean components

% ---------- For α-stable noise ----------
P_sig = 1;     % target signal power for SNR definition
delta = 0;     % location (mean) for noise

% ---------- quantum/Hamiltonian items ----------
hbar = 1; mass = 1;
const_Hamil = hbar^2 / (2*mass);
Hamil_P_base = Hamiltonian_Original(N_r, L, const_Hamil);

% ---------- Collectors ----------
err_y_NL = zeros(1, numel(SNR_Range));
err_y    = zeros(1, numel(SNR_Range));

err_x_LS_QD          = zeros(1, numel(SNR_Range));
err_x_LS             = zeros(1, numel(SNR_Range));
err_x_NL_LS_QD       = zeros(1, numel(SNR_Range));
err_x_VAMP_QD        = zeros(1, numel(SNR_Range));
err_x_VAMP_PlugPlay  = zeros(1, numel(SNR_Range));
err_x_VAMP_ori       = zeros(1, numel(SNR_Range));

% ===== NEW: Huber M-estimator baselines =====
err_x_Huber        = zeros(1, numel(SNR_Range));   % Huber on Y
err_x_Huber_QD     = zeros(1, numel(SNR_Range));   % Huber on Y_tilde

% Huber options (one place)
optsHuber.maxIter = 50;
optsHuber.tol     = 1e-6;
optsHuber.deltaScale = 1.345;   % delta = 1.345*sigma (standard Gaussian-efficient choice)



for iter = 1:numel(SNR_Range)
    SNR_dB   = SNR_Range(iter);
    var_noise = P_sig / 10^(SNR_dB/10);   % target noise var for alpha=2

    optsHuber.useFixedDelta = true;
    optsHuber.delta        = 1.5 * sqrt(var_noise);

    frame_valid_QD = 0;
    frame_valid_PP = 0;

    for frame = 1:FrameAll
        % ===== Noise: α-stable (colored optional) =====
        noise = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta);
        % noise = sqrt(var_noise) * randn(N_r,L);      % background
        % idx = rand(N_r,L) < 0.1;                     % 10% impulses
        % noise(idx) = noise(idx) + 50*randn(nnz(idx),1); % add large impulses

        % ===== Channel =====
        Channel_H = sqrt(var_channel) * randn(N_r, N_t);

        % ===== Tx Signal (all standardized to mean=0, var≈1) =====
        Symbol_X = make_signal(cfg, N_t, L);

        % ===== I/O =====
        Chan_Symbol = Channel_H * Symbol_X;
        Rec_Y       = Chan_Symbol + noise;
        Rec_vec_y   = Rec_Y(:);

        % ===== Denoise Y (external) =====
        [Rec_Y_NL_vec, tau, U_P] = DenoisedSignal_QD(Hamil_P_base, Rec_vec_y, Chan_Symbol);
        Rec_Y_NL = reshape(Rec_Y_NL_vec, N_r, L);

        % ===== Y NMSE (optional metrics) =====
        eNL = norm(Rec_Y_NL_vec - Chan_Symbol(:))^2 / max(norm(Chan_Symbol(:))^2, eps);
        eY  = norm(Rec_vec_y    - Chan_Symbol(:))^2 / max(norm(Chan_Symbol(:))^2, eps);
        if ~isfinite(eNL), eNL = 0; end
        if ~isfinite(eY ), eY  = 0; end
        err_y_NL(iter) = err_y_NL(iter) + eNL;
        err_y(iter)    = err_y(iter)    + eY;

        % ===== LS on Y / Y_tilde =====
        Hpinv  = pinv(Channel_H);
        LS_Ori = Hpinv * Rec_Y;
        LS_QD  = Hpinv * Rec_Y_NL;

        err_x_LS(iter)      = err_x_LS(iter)      + nmse(LS_Ori, Symbol_X);
        err_x_LS_QD(iter)   = err_x_LS_QD(iter)   + nmse(LS_QD,  Symbol_X);

        % ===== NEW: Huber M-estimator on Y / Y_tilde =====
        Huber_Ori = huber_gd(Channel_H, Rec_Y,    optsHuber);
        Huber_QD  = huber_gd(Channel_H, Rec_Y_NL, optsHuber);

        err_x_Huber(iter)    = err_x_Huber(iter)    + nmse(Huber_Ori, Symbol_X);
        err_x_Huber_QD(iter) = err_x_Huber_QD(iter) + nmse(Huber_QD,  Symbol_X);

        % ===== Nonlinear LS on Y_tilde (you provided QD_NL_LS) =====
        mat_A = U_P * diag(tau) * U_P' * kron(eye(L), Channel_H);
        Xhat_NL = QD_NL_LS(Channel_H, U_P, tau, Rec_Y_NL, Symbol_X);
        Xhat_NL = reshape(Xhat_NL, N_t, L);
        err_x_NL_LS_QD(iter) = err_x_NL_LS_QD(iter) + nmse(Xhat_NL, Symbol_X);

     
    end

    % ===== Average over frames =====
    err_y_NL(iter) = err_y_NL(iter) / FrameAll;
    err_y(iter)    = err_y(iter)    / FrameAll;

    err_x_LS(iter)        = err_x_LS(iter)        / FrameAll;
    err_x_LS_QD(iter)     = err_x_LS_QD(iter)     / FrameAll;
    err_x_Huber(iter)     = err_x_Huber(iter)     / FrameAll;      % NEW
    err_x_Huber_QD(iter)  = err_x_Huber_QD(iter)  / FrameAll;      % NEW

    err_x_NL_LS_QD(iter)  = err_x_NL_LS_QD(iter)  / FrameAll;
    % err_x_VAMP_ori(iter)  = err_x_VAMP_ori(iter)  / FrameAll;
    % 
    % if frame_valid_QD > 0
    %     err_x_VAMP_QD(iter) = err_x_VAMP_QD(iter) / frame_valid_QD;
    % else
    %     err_x_VAMP_QD(iter) = NaN;
    %     warning('All VAMP-QD frames invalid at SNR=%d dB', SNR_Range(iter));
    % end

    % if frame_valid_PP > 0
    %     err_x_VAMP_PlugPlay(iter) = err_x_VAMP_PlugPlay(iter) / frame_valid_PP;
    % else
    %     err_x_VAMP_PlugPlay(iter) = NaN;
    %     warning('All VAMP-PP frames invalid at SNR=%d dB', SNR_Range(iter));
    % end
end

% ===== Plot NMSE over SNR =====
% Auto title
noiseTypeStr = sprintf('$\\alpha$-stable noise ($\\alpha=%g$)', cfg.alpha);
if cfg.useColored
    noiseTypeStr = [noiseTypeStr, ', colored'];
else
    noiseTypeStr = [noiseTypeStr, ', white'];
end

if ~cfg.useColored
    noiseTypeStr = sprintf('%s, white', noiseTypeStr(1:strfind(noiseTypeStr,')')));
end

switch lower(cfg.sigModel)
    case 'gmm'
        piStr = sprintf('[%.2g, %.2g]', cfg.pi_mix(1), cfg.pi_mix(2));
        sigTypeStr = sprintf('GMM signal (%s)', piStr);
    case 'gaussian'
        sigTypeStr = 'Gaussian signal';
    case 'noprior'
        sigTypeStr = 'no-prior signal';
    otherwise
        sigTypeStr = 'unknown signal';
end

%% Figure 2: Error in X
figure
semilogy(SNR_Range, err_x_LS, '-o', ...
    'Color', [0.00, 0.45, 0.74], 'MarkerFaceColor', [0.00, 0.45, 0.74], 'LineWidth', 2); % 蓝
hold on 

% ===== NEW: Huber baselines =====
semilogy(SNR_Range, err_x_Huber, '-p', ...
    'Color', [0.20, 0.20, 0.20], 'MarkerFaceColor', [0.20, 0.20, 0.20], 'LineWidth', 2); % 黑灰
semilogy(SNR_Range, err_x_Huber_QD, '-h', ...
    'Color', [0.60, 0.60, 0.60], 'MarkerFaceColor', [0.60, 0.60, 0.60], 'LineWidth', 2); % 浅灰

semilogy(SNR_Range, err_x_LS_QD, '-s', ...
    'Color', [0.47, 0.67, 0.19], 'MarkerFaceColor', [0.47, 0.67, 0.19], 'LineWidth', 2); % 绿
semilogy(SNR_Range, err_x_NL_LS_QD, '-d', ...
    'Color', [0.49, 0.18, 0.56], 'MarkerFaceColor', [0.49, 0.18, 0.56], 'LineWidth', 2); % 紫 

grid on;
xlabel('SNR (dB)','Interpreter','latex');
ylabel('Estimation error in $\mathbf{X}$','Interpreter','latex');
legend({'LS on $\mathbf{Y}$',  ...
        'Huber on $\mathbf{Y}$', ...
        'Huber on $\tilde{\mathbf{Y}}$', ...
        'LS on $\tilde{\mathbf{Y}}$', 'Nonlinear LS on $\tilde{\mathbf{Y}}$'}, ...
        'Interpreter','latex','Location','best');

title(sprintf('%s, %s', noiseTypeStr, sigTypeStr),'Interpreter','latex');

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
            X = 3 + 10*randn(N_t, L);
        otherwise
            error('Unknown cfg.sigModel');
    end
end

function N = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta)
    % target var for alpha=2: var_noise = P_sig/10^(SNR/10)
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
                    if vr > 0
                        N(r,:) = N(r,:) * sqrt(var_noise / vr);
                    end
                end
            else
                N = N * (sqrt(var_noise) / std(N(:)));
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
        X = delta + sqrt(2)*gamma*randn(m,n); % Gaussian: N(delta, 2*gamma^2)
        return;
    end
    U = pi*(rand(m,n)-0.5);
    W = exprnd(1, m, n);
    X0 = gamma .* (sin(alpha.*U) ./ (cos(U).^(1/alpha))) .* ...
         ((cos((1-alpha).*U) ./ W).^((1-alpha)/alpha));
    X = X0 + delta;
end

function Xhat = huber_gd(H, Y, opts)
% huber_gd: minimize sum rho_delta(r_i) using gradient steps on Huber loss

    if ~isfield(opts,'maxIter'), opts.maxIter = 200; end
    if ~isfield(opts,'tol'),     opts.tol     = 1e-7; end
    if ~isfield(opts,'deltaScale'), opts.deltaScale = 1.0; end
    if ~isfield(opts,'step'),    opts.step    = 0.5; end  % step size

    [Nr, Nt] = size(H);
    L = size(Y,2);
    Xhat = zeros(Nt, L);

    for ell=1:L
        y = Y(:,ell);
        x = pinv(H)*y;  % init

        r0 = y - H*x;
        if isfield(opts,'useFixedDelta') && opts.useFixedDelta
            delta = opts.delta;
        else
            sigma = 1.4826 * median(abs(r0 - median(r0)) + eps);
            delta = opts.deltaScale * sigma;
        end

        for k=1:opts.maxIter
            r = y - H*x;
            % psi for Huber
            psi = r;
            idx = abs(r) > delta;
            psi(idx) = delta * sign(r(idx));

            % gradient: -H' psi
            g = -(H' * psi);

            x_new = x - opts.step * g;

            if norm(x_new-x)/(norm(x)+eps) < opts.tol
                x = x_new; break;
            end
            x = x_new;
        end

        Xhat(:,ell) = x;
    end
end

function [x_est, nmse_val] = QD_NL_LS(H, Psi, tau, Y_denoised, X_true)
    max_iter = 800;
    eta = 1e-3;
    % QD_NL_LS  Nonlinear LS estimation with quantum-inspired denoising

    [N_r, L] = size(Y_denoised);
    N_t = size(H, 2);

    % Vectorized form
    y_vec = Y_denoised(:);      % (N_r*L x 1)
    x0 = pinv(H) * Y_denoised;  % Initial LS guess
    x0 = x0(:);                 % Vectorized initial x (N_t*L x 1)

    % Precompute
    PsiT = Psi';                % (N_r x N_r)
    Tau = diag(tau);            % (N_r*L x N_r*L)
    H_big = kron(eye(L), H);    % (N_r*L x N_t*L)

    % Gradient: ∇J = -2 * (∂f/∂x)' * (y - f(x)) = -2 * A' * (y - A x)
    A = Psi * Tau * PsiT * H_big;

    % Define nonlinear function f(x) = Psi * Tau * Psi' * H_big * x
    f_x = @(x) A * x;

    % Initialize x
    x = x0;

    for it = 1:max_iter
        res = y_vec - f_x(x);         % Residual
        grad = -2 * (A') * res;       % Gradient
        x = x - eta * grad;           % Update step
    end

    % Output
    x_est = reshape(x, N_t, L);
    nmse_val = norm(x_est - X_true, 'fro')^2 / max(norm(X_true, 'fro')^2, eps);
end