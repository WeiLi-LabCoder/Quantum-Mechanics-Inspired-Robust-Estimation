%% White vs. Temporal-Colored Noise: PSD comparison (alpha=2 only)
clear; clc; rng(1);

% ---- parameters ----
N_r  = 1;                 % 单天线即可看时间PSD
L    = 16384;             % 时域长度要足够大以稳定PSD
fs   = 1;                 % 归一化采样率
SNRdB_ref = 0;            % 仅用于设定目标方差（无信号时：var_noise=1）
P_sig = 1;
var_noise = P_sig / 10^(SNRdB_ref/10);    % =1

% ---- Welch PSD params ----
nfft  = 4096;
win   = hann(2048);
nover = 1024;

% ---- config: alpha=2（高斯），先生成白噪声 ----
cfg.alpha = 2;
cfg.useColored = false;          % 先关着色
%（若只需要 temporal 对比，不用 colorMode 字段）
delta = 0;

% 函数需在路径中：make_noise_alpha_stable, sAS
N_white = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta);
xw = N_white(1,:);

% ---- 生成 temporal 有色噪声（与你代码一致的滤波器）----
cfg.useColored = true;
cfg.colorMode  = 'temporal';     % 关键：时间相关着色
cfg.h          = [0.5 1 0.5];    % 低通型（可改为 [1 -1] 看高通）
N_col = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta);
xc = N_col(1,:);

% ---- Welch PSD (dB/Hz) ----
[Pw,f]  = pwelch(xw, win, nover, nfft, fs, 'onesided');
[Pc,~]  = pwelch(xc, win, nover, nfft, fs, 'onesided');
Pw_dB = 10*log10(Pw);
Pc_dB = 10*log10(Pc);

% ---- Plot ----
figure; hold on; grid on;
plot(f, Pw_dB, '-o', 'Color',[0.00,0.45,0.74], 'MarkerFaceColor',[0.00,0.45,0.74], 'LineWidth',2, 'MarkerSize',5);
plot(f, Pc_dB, '-s', 'Color',[0.85,0.33,0.10], 'MarkerFaceColor',[0.85,0.33,0.10], 'LineWidth',2, 'MarkerSize',5);
xlabel('Normalized frequency ($f/f_s$)','Interpreter','latex');
ylabel('PSD (dB/Hz)','Interpreter','latex');
legend({'White noise','Temporal-colored noise'}, 'Interpreter','latex','Location','best');
title('PSD of White vs. Colored (Temporal) Gaussian Noise','Interpreter','latex');


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