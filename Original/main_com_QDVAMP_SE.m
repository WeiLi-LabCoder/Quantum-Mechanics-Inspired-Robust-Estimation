clear; clc;  rng(1); 

% ---------- Dimensions ----------
N_r = 1000; N_t = 1000; L = 1;
SNR_dB = 0; 
hbar = 1; mass = 1;
const_Hamil = hbar^2 / (2*mass);
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

[length_row, length_col] = size(measurement);
[U, S, V] = svd(measurement);
y_flag = (1/var_noise) * S' * U' * obser;

Hamil_P_base = Hamiltonian_Original(length_col, 1, const_Hamil);

% 若工作区没有 x_true（QD 去噪器的第三个参数），用真实信号填充一次
if ~exist('x_true','var'), x_true = Symbol_X; end

%% ===================== QD-VAMP (保持原样) =====================
gama_2(2) = 1 / pri_var;
x2_mean(:,2) = var_noise * (inv(measurement' * measurement * var_noise + gama_2(2) * eye(length_col))) * measurement' * obser;
 
x2_alpha = (gama_2(2) / length_col) * trace(inv(measurement' * measurement * var_noise + gama_2(2) * eye(length_col)));
x2_eta = gama_2(2) / x2_alpha; 
gama_1(2) = max(x2_eta - gama_2(2), 1e-6);
r1(:,2) = x2_mean(:,2) * x2_eta / gama_1(2);

T_NUM = 50; 
M_mc = 20;  % Monte Carlo 次数
FixedIter = 0;
t_last=[];
for t = 2:T_NUM
    % === Denoising module ===
    if t <= FixedIter
        x1_eta(t) = gama_1(t) + 1 / pri_var;
        x1_mean(:,t) = (r1(:,t) * gama_1(t) + pri_mean / pri_var) / x1_eta(t);
        
        gama_2(t) = x1_eta(t) - gama_1(t);
        r2(:,t) = (x1_mean(:,t) * x1_eta(t) - r1(:,t) * gama_1(t)) / gama_2(t);
    else
        x1_mean(:,t) = DenoisedSignal_QD(Hamil_P_base, r1(:,t), x_true);  

        % Numerical derivative to estimate divergence (eta) 
        epsilon = 1e-3;
        eta_list = zeros(1, M_mc); 
        for m = 1:M_mc
            temp_z = randn(size(r1(:,t)));
            g_r1_eps = DenoisedSignal_QD(Hamil_P_base, r1(:,t) + epsilon * temp_z, x_true);
            eta_list(m) = (temp_z' * (g_r1_eps - x1_mean(:,t))) / (epsilon * norm(temp_z)^2);
        end
        
        eta(t) = mean(eta_list);  % 平均值作为估计
        eta(t) = max(min(real(eta(t)), 0.95), 1e-3);  % 夹紧在合理范围 

        % Update
        norm_diff = norm(x1_mean(:,t) - r1(:,t))^2; 
        % gama_2(t) = 1 / (norm_diff / length_col); 
        denom = norm_diff / length_col + 1e-12;   % 数值保护
        gama_2(t) = 1 / denom;

        r2(:,t) = (x1_mean(:,t) - eta(t) * r1(:,t)) / (1 - eta(t));
    end

    % === LMMSE module ===
    D = inv((1/var_noise) * S' * S + gama_2(t) * eye(length_col));
    x2_mean(:,t) = V * D * (y_flag + gama_2(t) * V' * r2(:,t));
    
    alpha(t) = sum(diag(D) * gama_2(t)) / length_row;  % 你当前的写法
    alpha(t) = min(max(real(alpha(t)), 1e-6), 0.999);  % 仅数值防护，不改变方法

    r1(:,t+1) = (x2_mean(:,t) - alpha(t) * r2(:,t)) / (1 - alpha(t));
    
    x2_eta_inv(t) = alpha(t);
    gama_1(t+1) = gama_2(t) * (1 - x2_eta_inv(t)) / x2_eta_inv(t);
 
    post_mean = x2_mean(:,t);
    nmse_it(t) = norm(post_mean - Symbol_X,'fro')^2 / norm(Symbol_X,'fro')^2; 
end

%% =====================  SE  ===================== 
 
s  = diag(S); s2 = s.^2;
R  = numel(s2);
Z  = max(N_t - R, 0);
N  = N_t;
gamma_w = 1/var_noise;
 
switch lower(cfg.sigModel)
    case 'gaussian'
        var_x = pri_var;
    case 'gmm'
        % cfg.sigma_mix 是“标准差”，方差需平方
        var_x = sum(cfg.pi_mix .* (cfg.sigma_mix.^2));
    otherwise
        var_x = var(Symbol_X(:));
end

% ----- Module-2 的 α2 与 E2 谱形式 -----
alpha2_fun = @(g2) ( Z + sum( g2 ./ (gamma_w*s2 + g2) ) ) / N;
E2_fun     = @(g2,t2) (1/N)*( ...
                    (gamma_w^2*var_noise) * sum( s2 ./ (gamma_w*s2 + g2).^2 ) + ...
                    (g2.^2 .* t2) .* ( Z./(g2.^2) + sum( 1 ./ (gamma_w*s2 + g2).^2 ) ) );

% ----- SE 初始化 -----
K_SE   = T_NUM;
tau1_se   = zeros(K_SE,1);
nmse_se   = zeros(K_SE,1);
tau1_se(1)= var_x;
gamma1_se = 1/var_x;
epsd = 1e-18;

% MC 配置（可适当调大以更平滑）
Mz = 45;          % 对 z 的平均
Mdiv = 35;        % MC-divergence 重采样
eps_div = 1e-3;  % 微扰幅度

for k = 1:K_SE
    % ===== Module-1: 评估 alpha1 与 E1 =====
    if k <= FixedIter
        % 与代码中前 FixedIter 次使用的“高斯先验线性去噪器”一致
        a1 = (var_x*gamma1_se)/(1 + var_x*gamma1_se);
        alpha1 = a1;
        E1     = (a1-1)^2*var_x + (a1^2)*tau1_se(k);
    else
        % 使用与你的 QD 去噪器一致的 MC 评估（以真实 x_true 为条件）
        E1_acc = 0; alpha1_acc = 0;
        for m = 1:Mz
            z = randn(N,1);
            r = x_true + sqrt(tau1_se(k))*z;
            g_r = DenoisedSignal_QD(Hamil_P_base, r, x_true);
            E1_acc = E1_acc + (norm(g_r - x_true)^2)/N;

            % MC divergence （方向导数平均）
            div_acc = 0;
            for jj = 1:Mdiv
                v = randn(N,1);
                g_r_eps = DenoisedSignal_QD(Hamil_P_base, r + eps_div*v, x_true);
                div_acc = div_acc + (v'*(g_r_eps - g_r))/(eps_div*(v'*v));
            end
            alpha1_acc = alpha1_acc + div_acc/Mdiv;
        end
        E1     = E1_acc / Mz;
        alpha1 = max(min(real(alpha1_acc/Mz), 0.99), 1e-4);   % 数值夹紧
    end

    % τ2, γ2（SE 公式）
    tau2 = max( (E1 - alpha1^2 * tau1_se(k)) / max((1 - alpha1)^2, epsd), 0 );
    gamma2 = 1 / ( (1 - alpha1)^2 * (tau1_se(k) + tau2) );

    % ===== Module-2: α2, E2, τ1_next =====
    alpha2 = alpha2_fun(gamma2);
    E2     = E2_fun(gamma2, tau2);
    tau1_next = max( (E2 - alpha2^2 * tau2) / max((1 - alpha2)^2, epsd), 0 );

    % 记录并前推
    nmse_se(k) = tau1_se(k) / max(var_x, epsd);
    if k < K_SE, tau1_se(k+1) = tau1_next; end

    % γ1 的前推（若需要显示 γ1，可用：eta2 = gamma2/alpha2; gamma1_se = max(eta2-gamma2,epsd);）
    eta2 = gamma2 / max(alpha2, epsd);
    gamma1_se = max(eta2 - gamma2, epsd);
end

% 对齐长度
nmse_se = nmse_se(1:numel(nmse_it));

% ===== 对比图 =====
figure; hold on; grid on;
semilogy(2:numel(nmse_it), nmse_it(2:end), 'o-','LineWidth',1.6,'DisplayName','QD-VAMP (simulation)');
semilogy(2:numel(nmse_se), nmse_se(2:end), 's--','LineWidth',1.6,'DisplayName','SE (prediction, MC)');
xlabel('Iteration'); ylabel('NMSE');
title(sprintf('QD-VAMP vs SE  (M=%d, N=%d, SNR=%.1f dB)', N_r, N_t, SNR_dB));
legend('Location','best');

saveas(gcf, 'QDVAMP_GMM.fig');   

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
