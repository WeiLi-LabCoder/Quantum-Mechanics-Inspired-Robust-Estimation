clear all; clc

% Parameter setting
N_r = 6; N_t = 5; L = 1;
SNR_Range = 0:5:40;
FrameAll = 1;
var_channel = 3000;

% Hamiltonian
hbar = 1; mass = 1;
const_Hamil = hbar^2 / (2*mass);
Hamil_P_base = Hamiltonian_Original(N_r, L, const_Hamil);

% Set of control parameters (tau)
controlPara_Set = [ 0.05, 0];  % 可根据需要修改

% Preallocate
err_NL_all = zeros(length(controlPara_Set), length(SNR_Range));
err_ref_all = zeros(length(controlPara_Set), length(SNR_Range));  % ← 现在为矩阵

% Loop over control parameters
for cidx = 1:length(controlPara_Set)
    controlPara = controlPara_Set(cidx);
    err_NL = zeros(1, length(SNR_Range));
    err_ref = zeros(1, length(SNR_Range));  % 当前controlPara对应的ref误差

    for snr_idx = 1:length(SNR_Range)
        SNR_test = SNR_Range(snr_idx);
        var_noise = 10^(-0.1*SNR_test);

        for frame = 1:FrameAll
            % noise
            noise = poissrnd(var_noise, N_r, L) - var_noise;

            % correlated channel 
            rho = 0.99;
            H_vec = mvnrnd(zeros(N_r*N_t, 1), (1 - rho) * eye(N_r*N_t) + rho * ones(N_r*N_t));   
            Channel_H = reshape(H_vec, N_r, N_t); 

            % correlated signal 
            X_vec = mvnrnd(zeros(N_t*L,1), (1 - rho) * eye(N_t*L) + rho * ones(N_t*L));
            Symbol_X = reshape(X_vec, N_t, L); 

            % forward channel + noise
            Chan_Symbol = Channel_H * Symbol_X;
            Rec_Y = Chan_Symbol + noise;
            Rec_vec_y = Rec_Y(:);

            % quantum denoising (controlPara=0 means no interaction terms)
            Rec_Y_NL_vec = SignalDenoise(Rec_vec_y, Chan_Symbol, Hamil_P_base, controlPara);

            % NMSE
            err_NL(snr_idx) = err_NL(snr_idx) + norm(Rec_Y_NL_vec - Chan_Symbol(:))^2 / norm(Chan_Symbol(:))^2;
            err_ref(snr_idx) = err_ref(snr_idx) + norm(Rec_vec_y - Chan_Symbol(:))^2 / norm(Chan_Symbol(:))^2;
        end

        % average over frames
        err_NL(snr_idx) = err_NL(snr_idx) / FrameAll;
        err_ref(snr_idx) = err_ref(snr_idx) / FrameAll;
    end

    % save results for this tau
    err_NL_all(cidx, :) = err_NL;
    err_ref_all(cidx, :) = err_ref;
end

% === Plot ===
figure; 
color_list = lines(length(controlPara_Set));
for cidx = 1:length(controlPara_Set)
    semilogy(SNR_Range, err_NL_all(cidx,:), 'o-', 'Color', color_list(cidx,:), ...
        'DisplayName', ['$\tau=$', num2str(controlPara_Set(cidx))], 'LineWidth', 1.5);
    hold on;
end

% 画出tau = 0 的参考误差
idx_ref = find(controlPara_Set == 0, 1);
if ~isempty(idx_ref)
    semilogy(SNR_Range, err_ref_all(idx_ref, :), 'k--', 'DisplayName', 'No denoising', 'LineWidth', 1.5)
end

grid on
xlabel('SNR (dB)', 'Interpreter', 'latex')
ylabel('Estimation NMSE in $\mathbf{Y}$', 'Interpreter', 'latex')
legend('Interpreter','latex','Location','best')
title(['Quantum Denoising Performance ($N_r=$' num2str(N_r) ', $N_t=$' ...
    num2str(N_t) ', $L=$' num2str(L) ')'], 'Interpreter', 'latex')
