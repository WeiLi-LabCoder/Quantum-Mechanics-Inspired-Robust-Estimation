% The weight (tau) parameter is optimized in each frame 
% channel and transmitted signal varies in each frame 
clear all; clc
% Parameter setting
N_r = 16; N_t = 8; L = 1;  
SNR_Range = -5:5:20;
var_channel = 1;
FrameAll = 10;

hbar = 1; mass = 1;
const_Hamil = hbar^2 / (2*mass);
Hamil_P_base = Hamiltonian_Original(N_r, L, const_Hamil);

err_y_NL = zeros(1, length(SNR_Range));
err_y = zeros(1, length(SNR_Range));

err_x_LS_QD = zeros(1, length(SNR_Range));
err_x_LS = zeros(1, length(SNR_Range));
err_x_NL_LS_QD = zeros(1, length(SNR_Range));
err_x_NL_opt_LS_QD = zeros(1, length(SNR_Range));
err_x_VAMP_QD = zeros(1, length(SNR_Range));
err_x_VAMP_PlugPlay = zeros(1, length(SNR_Range));
err_x_VAMP_ori = zeros(1, length(SNR_Range));


beta_store = 1e10*ones(2,2);
MC_train = FrameAll; % training parameters within finite frames
for snr_idx = 1:length(SNR_Range)
    SNR_test = SNR_Range(snr_idx);
    var_noise = 10^(-0.1*SNR_test);
    MC_train_iter = 1;
    frame_valid_PlugPlay = 0;
    frame_valid_QD = 0;
    for frame = 1:FrameAll 
        %% Noise
        noise = sqrt(var_noise) * randn(N_r, L) ; % Gaussian noise 
        % noise = poissrnd(var_noise, N_r, L) - var_noise; % Poission noise
        % noise = sqrt(var_noise) * randn(N_r, L) + poissrnd(var_noise, N_r, L) - var_noise; % mixed Gaussian and Poission noise 

        % impulse_prob = 0.6;     
        % impulse_mag = 5 * sqrt(var_noise);   
        % impulse_mask = rand(N_r, L) < impulse_prob; 
        % noise_impulse = impulse_mag .* (2*randi([0, 1], N_r, L) - 1) .* impulse_mask; 
        % noise_gauss = sqrt(var_noise) * randn(N_r, L);
        % noise = noise_gauss + noise_impulse;

 
        % % alpha-stable noise
        % alpha = 1;
        % P_sig = 1; 
        % gamma = (P_sig ./ (10.^(SNR_test / 10))).^(1 / alpha);
        % noise = symmetric_alpha_stable(alpha, gamma, N_r, L);


        
        %% Channel setting
        % % correlated channel 
        % rho = 0.8; 
        % H_vec = mvnrnd(zeros(N_r*N_t, 1), (1 - rho) * eye(N_r*N_t) + rho * ones(N_r*N_t));   
        % Channel_H = reshape(H_vec, N_r, N_t); 

        Channel_H = sqrt(var_channel) * randn(N_r, N_t);   % i.i.d. channel
        
        % % low rank channel
        % H0 = sqrt(var_channel) * randn(N_r, N_t);
        % [U, S, V] = svd(H0, 'econ');
        % S(end, end) = 1e-4;  % 最小奇异值设得非常小
        % Channel_H = U * S * V';
        

        % Kronecker 
        % R_r = toeplitz(0.6.^(0:N_r-1));  % 接收端相关矩阵
        % R_t = toeplitz(0.6.^(0:N_t-1));  % 发射端相关矩阵
        % H_iid = sqrt(var_channel) * randn(N_r, N_t);
        % Channel_H = sqrtm(R_r) * H_iid * sqrtm(R_t);

        %% Signal setting
        % correlated signal 
        % X_vec = mvnrnd(zeros(N_t*L,1), (1 - rho) * eye(N_t*L) + rho * ones(N_t*L));
        % Symbol_X = reshape(X_vec, N_t, L); 
        
        Symbol_X = randn(N_t, L);  % i.i.d. signal
        % 
        % sparsity = 0.9;  
        % Symbol_X = (rand(N_t, L) < sparsity) .* randn(N_t, L); 

        % Symbol_X =  2*randi([0,1],N_t, L)-1; % binary signal

        % % Gaussian Mixture Model (GMM) singal 
        % pi_mix = [0.6, 0.4];       % 混合系数
        % sigma = [1, 10];            % 两个高斯分布的方差 
        % Symbol_X = zeros(N_t, L);
        % mask = rand(N_t, L) < pi_mix(2);  % 从高方差分布中采样的掩码
        % Symbol_X(mask) = sigma(2) * randn(sum(mask(:)), 1);
        % Symbol_X(~mask) = sigma(1) * randn(sum(~mask(:)), 1);


        % % nonlinear signal
        % sparsity = 0.1;                     
        % mask = rand(N_t, L) < sparsity;    
        % X_sparse = zeros(N_t, L);
        % X_sparse(mask) = randn(sum(mask(:)), 1);    
        % % === nonlinear mapping: Sigmoid ===
        % nonlinear_func = @(x) 1 ./ (1 + exp(-x));  % Sigmoid
        % Symbol_X = nonlinear_func(X_sparse);       

        %% Input-Output Relationship
        Chan_Symbol = Channel_H * Symbol_X;
        Rec_Y = Chan_Symbol + noise;
        Rec_vec_y = Rec_Y(:);

        %% Denoised received signal 
        [Rec_Y_NL_vec,tau,U_P] = DenoisedSignal_QD(Hamil_P_base,Rec_vec_y,Chan_Symbol);
        Rec_Y_NL = reshape(Rec_Y_NL_vec,N_r,L);

        % compute NMSE in each frame
        err_NL_temp = norm(Rec_Y_NL_vec - Chan_Symbol(:))^2 / norm(Chan_Symbol(:))^2;
        err_temp = norm(Rec_vec_y - Chan_Symbol(:))^2 / norm(Chan_Symbol(:))^2;
         if any(isnan( err_NL_temp)) || any(isinf(err_NL_temp))|| any(isnan( err_temp)) || any(isinf(err_temp))
            err_NL_temp=0; 
            err_temp=0;
         end
        err_y_NL(snr_idx) = err_y_NL(snr_idx) + err_NL_temp;
        err_y(snr_idx) = err_y(snr_idx) + err_temp; 

        %% Signal Recovery
        % LS estimation
        Channel_H_inv = pinv(Channel_H);
        LS_QD = Channel_H_inv * Rec_Y_NL;
        LS_Ori = Channel_H_inv * Rec_Y;

        err_x_LS_QD(snr_idx) = err_x_LS_QD(snr_idx) + norm(LS_QD - Symbol_X,'fro')^2 / norm(Symbol_X,'fro')^2;
        err_x_LS(snr_idx) = err_x_LS(snr_idx) + norm(LS_Ori - Symbol_X,'fro')^2 / norm(Symbol_X,'fro')^2;

        % nonlinear LS estimation 
        mat_A = U_P * diag(tau) * U_P'*kron(eye(L),Channel_H) ; 
        % NL_LS_QD = pinv(mat_A) * Rec_Y_NL_vec; 
        NL_LS_QD = QD_NL_LS(Channel_H, U_P, tau, Rec_Y_NL, Symbol_X);
        NL_LS_QD = reshape(NL_LS_QD,N_t,L);

        err_x_NL_LS_QD(snr_idx) = err_x_NL_LS_QD(snr_idx) + norm(NL_LS_QD -  Symbol_X,'fro')^2 / norm(Symbol_X,'fro')^2;

        % nonlinear robust LS estimation
        NL_opt_LS_QD = QD_NL_LS_opt(Channel_H, Rec_Y_NL, Symbol_X, tau);
        NL_opt_LS_QD = reshape(NL_opt_LS_QD,N_t,L);
        err_x_NL_opt_LS_QD(snr_idx) = err_x_NL_opt_LS_QD(snr_idx)+ norm(NL_opt_LS_QD -  Symbol_X,'fro')^2 / norm(Symbol_X,'fro')^2;
 
        % VAMP
        [est_X_VAMP_ori,~]=Recovery_VAMP(Rec_vec_y,kron(eye(L),Channel_H),0,1,var_noise);  
        err_x_VAMP_ori(snr_idx) = err_x_VAMP_ori(snr_idx) + ...
                norm(est_X_VAMP_ori - Symbol_X(:), 'fro')^2 / norm(Symbol_X, 'fro')^2; 


        % var_noise_new = norm(Rec_Y_NL - Chan_Symbol,'fro')^2/(N_r*L); 
        [est_X_VAMP_QD,~]=Recovery_VAMP(Rec_Y_NL_vec,mat_A,0,1,var_noise); 
        if ~any(isnan(est_X_VAMP_QD(:)))
            err_x_VAMP_QD(snr_idx) = err_x_VAMP_QD(snr_idx) + ...
                norm(est_X_VAMP_QD - Symbol_X(:), 'fro')^2 / norm(Symbol_X, 'fro')^2;
            frame_valid_QD = frame_valid_QD + 1;
        end 

        [est_X_VAMP_QD_PlugPlay ]=QD_VAMP(Rec_Y_NL_vec,mat_A,0,1,var_noise,Symbol_X(:),const_Hamil);
        if ~any(isnan(est_X_VAMP_QD_PlugPlay(:)))
            err_x_VAMP_PlugPlay(snr_idx) = err_x_VAMP_PlugPlay(snr_idx) + ...
                norm(est_X_VAMP_QD_PlugPlay - Symbol_X(:), 'fro')^2 / norm(Symbol_X, 'fro')^2;
            frame_valid_PlugPlay = frame_valid_PlugPlay + 1;
        end 

        
    end

    % NMSE in each SNR
    err_y_NL(snr_idx) = err_y_NL(snr_idx) / FrameAll;
    err_y(snr_idx) = err_y(snr_idx) / FrameAll;

    err_x_LS_QD(snr_idx) = err_x_LS_QD(snr_idx) / FrameAll;
    err_x_LS(snr_idx) = err_x_LS(snr_idx) / FrameAll;
    err_x_NL_LS_QD(snr_idx) = err_x_NL_LS_QD(snr_idx) / FrameAll;
    err_x_NL_opt_LS_QD(snr_idx) = err_x_NL_opt_LS_QD(snr_idx) / FrameAll;
    err_x_VAMP_ori(snr_idx) = err_x_VAMP_ori(snr_idx)/ FrameAll;
    if frame_valid_QD > 0
        err_x_VAMP_QD(snr_idx) = err_x_VAMP_QD(snr_idx) / frame_valid_QD;
    else
        warning("All VAMP frames invalid (NaN) at SNR = %d dB", SNR_test);
        err_x_VAMP_QD(snr_idx) = NaN;
    end

    if frame_valid_PlugPlay > 0
        err_x_VAMP_PlugPlay(snr_idx) = err_x_VAMP_PlugPlay(snr_idx) / frame_valid_PlugPlay;
    else
        warning("All VAMP frames invalid (NaN) at SNR = %d dB", SNR_test);
        err_x_VAMP_PlugPlay(snr_idx) = NaN;
    end

end

% % Plot
% figure
% semilogy(SNR_Range, err_y_NL, 'or-', 'LineWidth', 1.5)
% hold on
% semilogy(SNR_Range, err_y, 'xb-', 'LineWidth', 1.5)
% xlabel('SNR (dB)', 'Interpreter', 'latex')
% ylabel('Estimation error in $\mathbf{Y}$', 'Interpreter', 'latex')
% legend(['Quantum Denoising with optimal $\tau$ ($N_r=$',num2str(N_r),...
%     '$, N_t=$',num2str(N_t),'$, L=$',num2str(L),')'], ...
%     ['No denoising ($N_r=$',num2str(N_r),'$, N_t=$', ...
%     num2str(N_t),'$, L=$',num2str(L),')'], 'Interpreter', 'latex')
% grid on

figure 
semilogy(SNR_Range, err_x_LS, 'xb-', 'LineWidth', 1.5) 
hold on
semilogy(SNR_Range, err_x_VAMP_ori, '^m-', 'LineWidth', 1.5)
semilogy(SNR_Range, err_x_LS_QD, 'or-', 'LineWidth', 1.5) 
semilogy(SNR_Range, err_x_NL_LS_QD, '+k-', 'LineWidth', 1.5)
semilogy(SNR_Range, err_x_VAMP_QD, '>-','Color', [0.90, 0.43, 0.31], 'LineWidth', 1.5)
semilogy(SNR_Range, err_x_VAMP_PlugPlay, '<-', 'Color', [0.54, 0.69, 0.49], 'LineWidth', 1.5)
xlabel('SNR (dB)', 'Interpreter', 'latex')
ylabel('Estimation error in $\mathbf{X}$', 'Interpreter', 'latex')
legend(['LS on $\mathbf{Y}$ ($N_r=$',num2str(N_r),'$, N_t=$', ...
    num2str(N_t),'$, L=$',num2str(L),')'],...
    ['Original VAMP on ${\mathbf{Y}}$ ($N_r=$',num2str(N_r),'$, N_t=$', ...
    num2str(N_t),'$, L=$',num2str(L),')'],...
    ['LS on $\tilde{\mathbf{Y}}$ ($N_r=$',num2str(N_r),...
    '$, N_t=$',num2str(N_t),'$, L=$',num2str(L),')'], ... 
    ['Nonlinear LS on $\tilde{\mathbf{Y}}$ ($N_r=$',num2str(N_r),'$, N_t=$', ...
    num2str(N_t),'$, L=$',num2str(L),')'],... 
    ['VAMP on $\tilde{\mathbf{Y}}$ ($N_r=$',num2str(N_r),'$, N_t=$', ...
    num2str(N_t),'$, L=$',num2str(L),')'],...
    ['Plug-and-Play VAMP on $\tilde{\mathbf{Y}}$ ($N_r=$',num2str(N_r),'$, N_t=$', ...
    num2str(N_t),'$, L=$',num2str(L),')'],'Interpreter', 'latex')
grid on

title('Gaussian Noise, i.i.d. channel, i.i.d. signal','Interpreter', 'latex')

function [x_est, nmse_val] = QD_NL_LS(H, Psi, tau, Y_denoised, X_true)
max_iter = 800;
eta = 1e-3;
% QD_NL_LS  Nonlinear LS estimation with quantum-inspired denoising
 
    [N_r, L] = size(Y_denoised);
    N_t = size(H, 2);

    % Vectorized form
    y_vec = Y_denoised(:);      % (N_r*L x 1)
    x0 = pinv(H) * Y_denoised;  % Initial LS guess
    % x0 = zeros(N_t * L, 1);
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

    for iter = 1:max_iter
        res = y_vec - f_x(x);         % Residual
        grad = -2 * (A') * res;       % Gradient
        x = x - eta * grad;           % Update step 
    end

    % Output
    x_est = reshape(x, N_t, L);
    nmse_val = norm(x_est - X_true, 'fro')^2 / max(norm(X_true, 'fro')^2, eps);
end


function [X_est, nmse_val] = QD_NL_LS_opt(H, Y_denoised, X_true, tau)
max_iter=1000;
eta = 1e-3; 
% QD_NL_LS_opt_GD  Estimate signal X from denoised observation Y_denoised
% using gradient descent with residual weighting (nonlinear LS)

    % Dimensions
    [N_r, N_t] = size(H);
    [~, L] = size(Y_denoised);
    Y_vec = Y_denoised(:);  % Vectorized target
    H_big = kron(eye(L), H);  % Expand H to match full vector

    % Initialization: LS solution as warm start
    X_ls = pinv(H) * Y_denoised;
    x = X_ls(:);

    % Precompute diag(tau.^2)
    Tau2 = diag(tau(:).^2);

    for iter = 1:max_iter
        res = Y_vec - H_big * x;
        grad = -2 * H_big' * Tau2 * res;
        x = x - eta * grad;

        % Optional: early stopping
        if norm(grad) < 1e-6
            break;
        end
    end

    % Final estimate
    X_est = reshape(x, N_t, L);
    nmse_val = norm(X_est - X_true, 'fro')^2 / max(eps, norm(X_true, 'fro')^2);
end

function X = symmetric_alpha_stable(alpha, gamma, m, n)
% 生成对称 α-stable 噪声 (Chambers-Mallows-Stuck 方法)
% alpha ∈ (0, 2), gamma 是尺度参数

    if alpha <= 0 || alpha > 2
        error('alpha must be in the range (0, 2].');
    end

    % 步骤1：生成均匀分布和指数分布变量
    U = pi * (rand(m, n) - 0.5);  % Uniform(-π/2, π/2)
    W = exprnd(1, m, n);          % Exponential(1)

    % 步骤2：生成对称 alpha-stable 噪声
    X = gamma * (sin(alpha * U) ./ (cos(U).^(1/alpha))) .* ...
        ((cos((1 - alpha) * U) ./ W).^((1 - alpha) / alpha));
end


% function [x_est, nmse_val] = QD_NL_LS(H, Psi, tau, Y_denoised, X_true)
% % QD_NL_LS_fmincon  Nonlinear LS estimation with quantum-inspired denoising
% % using fmincon optimizer instead of manual gradient descent.
%     [N_r, L] = size(Y_denoised);
%     N_t = size(H, 2);
% 
%     y_vec = Y_denoised(:);  % Vectorized target
%     H_big = kron(eye(L), H);
%     Tau = diag(tau);
%     A = Psi * Tau * Psi' * H_big;
% 
%     % Initial guess: LS solution
%     x0 = pinv(H) * Y_denoised;
%     x0 = x0(:);  % Vectorized
% 
%     % Objective function
%     loss_fn = @(xvec) norm(A * xvec - y_vec)^2;
% 
%     % Optional bounds (can be omitted or modified)
%     lb = -ones(size(x0));
%     ub = ones(size(x0));
% 
%     % fmincon options
%     opts = optimoptions('fmincon', 'Display', 'off', ...
%         'MaxIter', 300, 'Algorithm', 'interior-point');
% 
%     % Solve optimization
%     x_opt = fmincon(loss_fn, x0, [], [], [], [], lb, ub, [], opts);
% 
%     % Output
%     x_est = reshape(x_opt, N_t, L);
%     nmse_val = norm(x_est - X_true, 'fro')^2 / max(eps, norm(X_true, 'fro')^2);
% end



% function [X_est, nmse_val] = QD_NL_LS_opt(H, Y_denoised, X_true, tau)
% % QD_NL_LS  Estimate signal X from denoised observation Y_denoised 
% % Dimensions
% [N_r, N_t] = size(H);
% [~, L] = size(Y_denoised);
% 
% % Initial guess using LS
% X_ls = pinv(H) * Y_denoised;
% x0 = X_ls(:);
% 
% % Loss function with soft weight tau
% Y_vec = Y_denoised(:);  % vectorized denoised Y
% loss_fn = @(xvec) norm(tau .* (Y_vec - reshape(H * reshape(xvec, N_t, L), [], 1)))^2;
% 
% % Bound constraints (optional)
% lb = -ones(size(x0));
% ub = ones(size(x0));
% 
% % Optimization
% opts = optimoptions('fmincon', 'Display', 'off', 'MaxIter', 300, 'Algorithm', 'interior-point');
% X_opt_vec = fmincon(loss_fn, x0, [], [], [], [], lb, ub, [], opts);
% X_est = reshape(X_opt_vec, N_t, L);
% 
% % Compute NMSE
% nmse_val = norm(X_est - X_true)^2 / max(eps, norm(X_true)^2);
% end




 