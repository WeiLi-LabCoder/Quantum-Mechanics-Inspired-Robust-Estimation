function [post_mean] = QD_VAMP_QAM_aware_trained(obser, measurement, pri_mean, pri_var, var_noise, const_Hamil, Mconst,beta_opt)
% QD_VAMP (Oracle) Plug-and-Play VAMP with Quantum Denoiser
% Robustifications:
%   (1) Complex-safe MC divergence: z ~ CN(0, I)
%   (2) epsilon scaled to ||r|| to avoid numerical instability
%   (2) alpha normalization uses length_col (N) rather than length_row (M)
%   (4) Constellation-aware soft projection (disabled by default)

[length_row, length_col] = size(measurement);

% Economy SVD for stability/speed
[U, S, V] = svd(measurement, 'econ');

s2 = diag(S).^2;  % s_i^2
s2 = max(s2, 1e-12 * max(s2));
 

% Precompute matched filter term  
y_flag = (1/var_noise) * (S') * (U') * obser;

% Hamiltonian base (length_col corresponds to x dimension)
Hamil_P_base = Hamiltonian_Original(length_col, 1, const_Hamil);

lambda = 0.2;           % soft projection strength in [0,1]

%% Initialization  
T_NUM = 15;

gama_1 = zeros(1, T_NUM+5);
gama_2 = zeros(1, T_NUM+5);
eta    = zeros(1, T_NUM+5);
alpha  = zeros(1, T_NUM+5);

x1_mean = zeros(length_col, T_NUM+5);
x2_mean = zeros(length_col, T_NUM+5);
r1      = zeros(length_col, T_NUM+5);
r2      = zeros(length_col, T_NUM+5);

% Initial precision
gama_2(2) = 1 / pri_var;

% Initial LMMSE estimate 
x2_mean(:,2) = var_noise * ((measurement' * measurement * var_noise + gama_2(2) * eye(length_col)) \ (measurement' * obser));
 
invMat = (measurement' * measurement * var_noise + gama_2(2) * eye(length_col));
x2_alpha = (gama_2(2)/length_col) * trace(invMat \ eye(length_col));    % normalize by N
x2_eta   = gama_2(2) / max(x2_alpha, eps);

gama_1(2) = max(x2_eta - gama_2(2), 1e-6);
r1(:,2)   = x2_mean(:,2) * x2_eta / gama_1(2);

%% Iteration
FixedIter = 1;
t_last = 2;

for t = 2:T_NUM
    % === Denoising module ===
    if t <= FixedIter
        x1_eta = gama_1(t) + 1 / pri_var;
        x1_mean(:,t) = (r1(:,t) * gama_1(t) + pri_mean / pri_var) / x1_eta;
        gama_2(t) = x1_eta - gama_1(t);
        r2(:,t)   = (x1_mean(:,t) * x1_eta - r1(:,t) * gama_1(t)) / gama_2(t);
    else
        % --------- Oracle quantum denoiser ---------  
        x1_mean(:,t) = DenoisedSignal_QD_trained(Hamil_P_base, r1(:,t), beta_opt); 
        % bad = ~isfinite(x1_mean(:,t));              
        % if any(bad(:))                              
        %     fprintf('Non-finite in x1_mean at t=%d, count=%d\n', t, nnz(bad));
        %     keyboard
        % end
        % constellation-aware soft projection
        xq = qam_project(x1_mean(:,t), Mconst);
        x1_mean(:,t) = (1-lambda)*x1_mean(:,t) + lambda*xq;
        % --------- MC divergence to estimate divergence (eta) (complex-safe) ---------
        eps0 = 1e-3;
        epsilon = eps0 * max(1, norm(r1(:,t))/sqrt(length_col));
        M_mc = 20;
        eta_list = zeros(1, M_mc);

        for m = 1:M_mc
            temp_z = (randn(size(r1(:,t))) + 1j*randn(size(r1(:,t))))/sqrt(2);
            g_r1_eps = DenoisedSignal_QD_trained(Hamil_P_base, r1(:,t) + epsilon * temp_z, beta_opt);
            % bad = ~isfinite(g_r1_eps);               
            % if any(bad(:))                              
            %     fprintf('Non-finite in x1_mean at t=%d, count=%d\n', t, nnz(bad));
            %     keyboard
            % end
            g_r1_eps = (1-lambda)*g_r1_eps + lambda*qam_project(g_r1_eps, Mconst);
            
            denom = epsilon * (temp_z' * temp_z);
            eta_list(m) = (temp_z' * (g_r1_eps - x1_mean(:,t))) / max(denom, eps);
        end

        eta(t) = mean(eta_list);
        eta(t) = max(min(real(eta(t)), 0.95), 1e-3);

        % Update
        norm_diff = norm(x1_mean(:,t) - r1(:,t))^2;
        if norm_diff < 1e-10 * norm(r1(:,t))^2 || norm_diff < 1e-14
            t_last = t;
            break;
        end

        gama_2(t) = 1 / max(norm_diff / length_col, 1e-12);
        r2(:,t)   = (x1_mean(:,t) - eta(t) * r1(:,t)) / (1 - eta(t));
    end

    % === LMMSE module (SVD form) ===   
    tmp = (V') * r2(:,t);
    tmp2 = y_flag + gama_2(t) * tmp;
    d  = 1 ./ ((1/var_noise) * s2 + gama_2(t));
    x2_mean(:,t) = V * (d .* tmp2);

    % ---- alpha normalized by N ----  
    alpha(t) = (gama_2(t)/length_col) * sum(d);
    alpha(t) = min(max(alpha(t), 1e-6), 1-1e-6);

    r1(:,t+1) = (x2_mean(:,t) - alpha(t) * r2(:,t)) / (1 - alpha(t));

    gama_1(t+1) = gama_2(t) * (1 - alpha(t)) / max(alpha(t), 1e-12);

    % stopping
    if norm(r1(:,t+1)-r1(:,t))^2 < max(norm(r1(:,t))^2, eps) * 1e-12
        t_last = t;
        break;
    end
    t_last = t;
end

post_mean = x2_mean(:,t_last);

end

%% ===== QAM projection =====
function xq = qam_project(x, M)
    idx = qamdemod(x, M, 'UnitAveragePower', true);
    xq  = qammod(idx, M, 'UnitAveragePower', true);
end