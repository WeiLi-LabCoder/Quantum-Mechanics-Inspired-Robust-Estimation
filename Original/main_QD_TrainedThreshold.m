clear all; clc

% parameter setting
N_r = 4; % number of antennas at the receiver 
N_t = 4; % number of antennas at the transmitter 
L = 4; % number of time slots in transmitted signal  
SNR_Range = -20:5:10;
var_channel = 1; % variance of the channel 
FrameAll = 3000; MC_train = 1000;
hbar = 1; mass = 1; const_Hamil = hbar^2 / (2*mass);
Hamil_P_base = Hamiltonian_Original(N_r, L, const_Hamil); 

err_NL = zeros(1, length(SNR_Range));
err = zeros(1, length(SNR_Range));
beta_SNR = zeros(2, length(SNR_Range));

% fixed channel 
rng(42); 
% correlated channel 
% rho = 0.99; 
% H_vec = mvnrnd(zeros(N_r*N_t, 1), (1 - rho) * eye(N_r*N_t) + rho * ones(N_r*N_t));   
% Channel_H = reshape(H_vec, N_r, N_t); 
Channel_H = sqrt(var_channel) * randn(N_r, N_t);   % i.i.d. channel
rng('shuffle');


%% ------------------- training parameters (beta) ------------------- %%
for snr_idx = 1:length(SNR_Range)
    SNR = SNR_Range(snr_idx);
    var_noise = 10^(-0.1 * SNR);

    U_P_list = cell(MC_train,1);
    alpha_train_list = cell(MC_train,1);
    alpha_true_list = cell(MC_train,1);

    for frame = 1:MC_train
        % noise
        % noise = sqrt(var_noise) * randn(N_r, L) ; % Gaussian noise 
        noise = poissrnd(var_noise, N_r, L) - var_noise; % Poission noise
        % noise = sqrt(var_noise) * randn(N_r, L) + poissrnd(var_noise, N_r, L) - var_noise; % mixed Gaussian and Poission noise 
 
        % correlated signal 
        % X_vec = mvnrnd(zeros(N_t*L,1), (1 - rho) * eye(N_t*L) + rho * ones(N_t*L));
        % Symbol_X = reshape(X_vec, N_t, L); 
        % Symbol_X = randn(N_t, L);  % i.i.d. signal
        Symbol_X = randi([0,1],N_t, L); % binary signal 
        
        Chan_Symbol = Channel_H * Symbol_X;
        Rec_vec_y = Chan_Symbol(:) + noise(:);

        Hamil_P = Hamil_P_base + diag(Rec_vec_y);
        [U_P, ~, ~] = svd(Hamil_P);
        alpha = linsolve(U_P, Rec_vec_y);
        alpha_true = linsolve(U_P, Chan_Symbol(:));

        U_P_list{frame} = U_P;
        alpha_train_list{frame} = alpha;
        alpha_true_list{frame} = alpha_true;
    end

    % optimize beta 
    beta_init = [0.6; 0.01];
    options = optimset('Display','off','MaxIter',500);
    beta_opt = fminsearch(@(b) ...
        sum(arrayfun(@(i) ...
            norm(U_P_list{i} * (alpha_train_list{i} .* ...
            (1./(1 + exp(b(1)*(abs(alpha_train_list{i}) - b(2))))) ) ...
            - U_P_list{i} * alpha_true_list{i})^2, ...
            1:MC_train)), ...
        beta_init, options);

    beta_SNR(:, snr_idx) = beta_opt;
end

%% ------------------- Test ------------------- %%
for snr_idx = 1:length(SNR_Range)
    SNR = SNR_Range(snr_idx);
    var_noise = 10^(-0.1 * SNR);
    beta_opt = beta_SNR(:, snr_idx);

    for frame = 1:FrameAll
        % noise
        % noise = sqrt(var_noise) * randn(N_r, L) ; % Gaussian noise 
        noise = poissrnd(var_noise, N_r, L) - var_noise; % Poission noise
        % noise = sqrt(var_noise) * randn(N_r, L) + poissrnd(var_noise, N_r, L) - var_noise; % mixed Gaussian and Poission noise 

        % correlated signal 
        % X_vec = mvnrnd(zeros(N_t*L,1), (1 - rho) * eye(N_t*L) + rho * ones(N_t*L));
        % Symbol_X = reshape(X_vec, N_t, L); 
        Symbol_X = randn(N_t, L);  % i.i.d. signal
        % Symbol_X = randi([0,1],N_t, L); % binary signal 

        Chan_Symbol = Channel_H * Symbol_X;
        Rec_vec_y = Chan_Symbol(:) + noise(:);

        Hamil_P = Hamil_P_base + diag(Rec_vec_y);
        [U_P, ~, ~] = svd(Hamil_P);
        alpha = linsolve(U_P, Rec_vec_y);
        tau = 1 ./ (1 + exp(beta_opt(1) * (abs(alpha) - beta_opt(2))));
        Rec_Y_NL_vec = U_P * (alpha .* tau);

        err_NL_temp = norm(Rec_Y_NL_vec - Chan_Symbol(:))^2 / norm(Chan_Symbol(:))^2;
        err_temp = norm(Rec_vec_y - Chan_Symbol(:))^2 / norm(Chan_Symbol(:))^2;
         if any(isnan( err_NL_temp)) || any(isinf(err_NL_temp))|| any(isnan( err_temp)) || any(isinf(err_temp))
            err_NL_temp=0; 
            err_temp=0;
         end
         err_NL(snr_idx) = err_NL(snr_idx) + err_NL_temp;
        err(snr_idx) = err(snr_idx) + err_temp; 
    end

    err_NL(snr_idx) = err_NL(snr_idx) / FrameAll;
    err(snr_idx) = err(snr_idx) / FrameAll;
end

%% ------------------- Plot ------------------- %%
figure
semilogy(SNR_Range, err_NL, 'or-', 'LineWidth', 1.5); hold on
semilogy(SNR_Range, err, 'xb-', 'LineWidth', 1.5);
xlabel('SNR (dB)', 'Interpreter', 'latex')
ylabel('Estimation error in $\mathbf{Y}$', 'Interpreter', 'latex')
legend(['Quantum Denoising with trained $\tau$ ($N_r=$',num2str(N_r),...
    '$, N_t=$',num2str(N_t),'$, L=$',num2str(L),')'], ...
    ['No denoising ($N_r=$',num2str(N_r),'$, N_t=$', ...
    num2str(N_t),'$, L=$',num2str(L),')'], 'Interpreter', 'latex')
grid on
