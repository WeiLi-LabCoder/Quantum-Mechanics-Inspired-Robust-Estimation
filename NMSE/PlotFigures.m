% 载入数据
data = load('result_qaoa_ls_N_r8N_t8L4_layer3.mat');

% === 图 1: 去噪对比 ===
figure;
semilogy(data.SNR_Range, data.err, 'xb-', 'LineWidth', 1.8); hold on;
semilogy(data.SNR_Range, data.err_NL, 'or-', 'LineWidth', 1.8);
xlabel('SNR (dB)', 'Interpreter', 'latex');
ylabel('Estimation error', 'Interpreter', 'latex');
title('Quantum denoising vs. No denoising', 'Interpreter', 'latex');
grid on;
legend('No denoising', 'Quantum denoising only', 'Interpreter', 'latex');
set(gca, 'FontSize', 12);

% === 图 2: 各估计方法比较 ===
figure;
semilogy(data.SNR_Range, data.err_LS, 'k-o', 'LineWidth', 1.5); hold on;
semilogy(data.SNR_Range, data.err_QAOA, 'k--*', 'LineWidth', 1.5);
semilogy(data.SNR_Range, data.err_LS_NL, 'b-.^', 'LineWidth', 1.5);
semilogy(data.SNR_Range, data.err_QAOA_NL, 'm:v', 'LineWidth', 1.5);
semilogy(data.SNR_Range, data.err_MMSE, '->', 'Color', [0.54, 0.69, 0.49],'LineWidth', 1.5);
semilogy(data.SNR_Range, data.err_QD_NL_LS, 'r-s', 'LineWidth', 1.5); 

% semilogy(data.SNR_Range, data.err_MMSE_NL, 'm:', 'LineWidth', 1.5);
% semilogy(data.SNR_Range, data.err_QD_NL_LS_QAOA, 'b-x', 'LineWidth', 1.5);

xlabel('SNR (dB)', 'Interpreter', 'latex');
ylabel('Recovery NMSE', 'Interpreter', 'latex');
title('Signal Estimation: Raw vs. Denoised + LS/QAOA', 'Interpreter', 'latex');
grid on;
legend('LS on $\mathbf{Y}$', 'QAOA on $\mathbf{Y}$', 'LS on $\tilde{\mathbf{Y}}$', 'QAOA on $\tilde{\mathbf{Y}}$', ...
       'MMSE on $\mathbf{Y}$','Nonlinear LS on $\tilde{\mathbf{Y}}$', 'Interpreter', 'latex');  % 若添加更多方法，更新 legend
set(gca, 'FontSize', 12);
