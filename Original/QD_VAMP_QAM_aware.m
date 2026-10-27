function [post_mean] = QD_VAMP_QAM_aware(obser, measurement, pri_mean, pri_var, var_noise, x_true, const_Hamil, Mconst)
% QD_VAMP_QAM_aware
% Plug-and-Play VAMP with Quantum Denoiser + (optional) QAM-aware soft projection
%
% Robustifications added:
%   (0) Input sanity checks for NaN/Inf and invalid variances
%   (1) Avoid explicit inv(); use A\I instead
%   (2) Small Tikhonov regularization in linear solves
%   (3) Complex-safe MC divergence; skip NaN samples
%   (4) Guard qamdemod/qammod against NaN/Inf (projection fallback)
%   (5) Clamp alpha/eta/gammas to safe ranges

% -------------------- Sanity checks --------------------
[length_row, length_col] = size(measurement);

% default safe output
post_mean = zeros(length_col, 1);

if isempty(obser) || isempty(measurement)
    warning('QD_VAMP_QAM_aware: empty input.');
    return;
end

if any(~isfinite(obser(:)))
    warning('QD_VAMP_QAM_aware: obser has NaN/Inf. Return zeros.');
    return;
end
if any(~isfinite(measurement(:)))
    warning('QD_VAMP_QAM_aware: measurement has NaN/Inf. Return zeros.');
    return;
end

if ~isfinite(var_noise) || var_noise <= 0
    warning('QD_VAMP_QAM_aware: var_noise invalid (%g). Clamp to 1e-12.', var_noise);
    var_noise = 1e-12;
end
if ~isfinite(pri_var) || pri_var <= 0
    warning('QD_VAMP_QAM_aware: pri_var invalid (%g). Clamp to 1.', pri_var);
    pri_var = 1;
end
if ~isfinite(pri_mean)
    pri_mean = 0;
end

% -------------------- SVD (econ) --------------------
% (If measurement is huge / ill-conditioned, this is still the most stable path.)
[U, S, V] = svd(measurement, 'econ');

% matched filter term
y_flag = (1/var_noise) * (S') * (U') * obser;

% Hamiltonian base (length_col corresponds to x dimension)
Hamil_P_base = Hamiltonian_Original(length_col, 1, const_Hamil);

% QAM projection strength
lambda = 0.2;  % in [0,1]

% -------------------- Iteration parameters --------------------
T_NUM = 20;
FixedIter = 1;

% robust knobs
reg0   = 1e-12;   % baseline regularization
epsGam = 1e-12;   % clamp for precisions
epsDen = 1e-12;   % clamp for denominators
alphaMax = 0.999; % keep < 1 to avoid division blow-up

% MC divergence settings
M_mc = 20;
eps0 = 1e-3;

% -------------------- State init --------------------
gama_1 = zeros(1, T_NUM+5);
gama_2 = zeros(1, T_NUM+5);
eta    = zeros(1, T_NUM+5);
alpha  = zeros(1, T_NUM+5);

x1_mean = zeros(length_col, T_NUM+5);
x2_mean = zeros(length_col, T_NUM+5);
r1      = zeros(length_col, T_NUM+5);
r2      = zeros(length_col, T_NUM+5);

% Initial precision
gama_2(2) = max(1 / pri_var, epsGam);

% Initial LMMSE estimate (more stable linear solve)
A0 = (measurement' * measurement) * var_noise + gama_2(2) * eye(length_col);
A0 = A0 + reg0 * eye(length_col);

x2_mean(:,2) = var_noise * (A0 \ (measurement' * obser));

% Compute x2_alpha via trace(A0^{-1}) without inv()
trA0inv = trace(A0 \ eye(length_col));
x2_alpha = (gama_2(2) / length_col) * trA0inv;
x2_alpha = min(max(real(x2_alpha), epsDen), alphaMax);

x2_eta   = gama_2(2) / max(x2_alpha, epsDen);

gama_1(2) = max(x2_eta - gama_2(2), 1e-6);
r1(:,2)   = x2_mean(:,2) * x2_eta / max(gama_1(2), epsGam);

t_last = 2;

% -------------------- Main loop --------------------
for t = 2:T_NUM

    % ===== Denoising module =====
    if t <= FixedIter
        x1_eta = gama_1(t) + 1 / pri_var;
        x1_eta = max(x1_eta, epsGam);

        x1_mean(:,t) = (r1(:,t) * gama_1(t) + pri_mean / pri_var) / x1_eta;

        gama_2(t) = max(x1_eta - gama_1(t), epsGam);
        r2(:,t)   = (x1_mean(:,t) * x1_eta - r1(:,t) * gama_1(t)) / max(gama_2(t), epsGam);

    else
        % ---- Oracle quantum denoiser ----
        x1_mean(:,t) = DenoisedSignal_QD(Hamil_P_base, r1(:,t), x_true);

        % guard denoiser output
        if any(~isfinite(x1_mean(:,t)))
            warning('QD_VAMP_QAM_aware: denoiser produced NaN/Inf at t=%d. Fallback to r1.', t);
            x1_mean(:,t) = r1(:,t);
        end

        % ---- constellation-aware soft projection (safe) ----
        if ~isempty(Mconst)
            xq = qam_project_safe(x1_mean(:,t), Mconst);
            x1_mean(:,t) = (1-lambda)*x1_mean(:,t) + lambda*xq;
        end

        % ---- MC divergence (complex-safe) ----
        epsilon = eps0 * max(1, norm(r1(:,t))/sqrt(length_col));
        eta_acc = 0;
        cnt = 0;

        for m = 1:M_mc
            temp_z = (randn(size(r1(:,t))) + 1j*randn(size(r1(:,t))))/sqrt(2);

            g_r1_eps = DenoisedSignal_QD(Hamil_P_base, r1(:,t) + epsilon * temp_z, x_true);

            if any(~isfinite(g_r1_eps(:)))
                continue; % skip bad sample
            end

            if ~isempty(Mconst)
                g_r1_eps = (1-lambda)*g_r1_eps + lambda*qam_project_safe(g_r1_eps, Mconst);
            end

            denom = epsilon * (temp_z' * temp_z);
            denom = max(real(denom), epsDen);

            val = (temp_z' * (g_r1_eps - x1_mean(:,t))) / denom;

            if isfinite(val)
                eta_acc = eta_acc + val;
                cnt = cnt + 1;
            end
        end

        if cnt == 0
            eta(t) = 0.1; % fallback
        else
            eta(t) = eta_acc / cnt;
        end

        % clamp eta
        eta(t) = real(eta(t));
        eta(t) = max(min(eta(t), 0.95), 1e-3);

        % Update gamma2 using residual energy
        norm_diff = norm(x1_mean(:,t) - r1(:,t))^2;
        denom_nd  = max(norm_diff / length_col, 1e-12);
        gama_2(t) = max(1 / denom_nd, epsGam);

        % Update r2 safely
        r2(:,t) = (x1_mean(:,t) - eta(t) * r1(:,t)) / max(1 - eta(t), 1e-6);

        % optional stopping (avoid 0/0)
        if norm_diff < 1e-10 * max(norm(r1(:,t))^2, 1) || norm_diff < 1e-14
            t_last = t;
            break;
        end
    end

    % ===== LMMSE module (SVD form) =====
    % D = inv((1/var_noise) * S'S + gama2 I) but computed stably as solve
    A = (1/var_noise) * (S' * S) + gama_2(t) * eye(length_col);
    A = A + reg0 * eye(length_col);

    D = A \ eye(length_col);

    x2_mean(:,t) = V * D * (y_flag + gama_2(t) * (V') * r2(:,t));

    % alpha normalized by N
    alpha(t) = (gama_2(t)/length_col) * trace(D);
    alpha(t) = real(alpha(t));
    alpha(t) = min(max(alpha(t), epsDen), alphaMax);

    % update r1
    r1(:,t+1) = (x2_mean(:,t) - alpha(t) * r2(:,t)) / max(1 - alpha(t), 1e-6);

    % update gamma1
    gama_1(t+1) = gama_2(t) * (1 - alpha(t)) / max(alpha(t), epsDen);
    gama_1(t+1) = max(gama_1(t+1), 1e-6);

    % stopping
    if norm(r1(:,t+1)-r1(:,t))^2 < max(norm(r1(:,t))^2, eps) * 1e-12
        t_last = t;
        break;
    end

    % guard any NaN explosion
    if any(~isfinite(r1(:,t+1))) || any(~isfinite(x2_mean(:,t)))
        warning('QD_VAMP_QAM_aware: NaN/Inf detected at t=%d. Stop early.', t);
        t_last = t;
        break;
    end

    t_last = t;
end

post_mean = x2_mean(:,t_last);

% final guard
if any(~isfinite(post_mean))
    warning('QD_VAMP_QAM_aware: post_mean NaN/Inf, return zeros.');
    post_mean = zeros(length_col,1);
end

end


% =====================================================================
% Safe QAM projection (guard qamdemod/qammod)
% =====================================================================
function xq = qam_project_safe(x, M)
    xq = x;

    if isempty(M) || ~isscalar(M) || ~isfinite(M) || M < 2
        return;
    end

    if any(~isfinite(x(:)))
        % skip projection if invalid
        return;
    end

    try
        idx = qamdemod(x, M, 'UnitAveragePower', true);
        xq  = qammod(idx, M, 'UnitAveragePower', true);
    catch
        % if toolbox complains (rare), just skip projection
        xq = x;
    end
end