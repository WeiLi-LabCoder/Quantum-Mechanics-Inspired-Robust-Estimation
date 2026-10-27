%% ------------------- training parameters (beta) ------------------- %% 
function beta_SNR = ParaTrain(SNR_Range,MC_train,cfg,N_r, N_t, L, var_channel,P_sig, delta,Hamil_P_base)
for snr_idx = 1:length(SNR_Range) 
    SNR_dB    = SNR_Range(snr_idx);
    var_noise = P_sig / 10^(SNR_dB/10);

    U_P_list = cell(MC_train,1);
    alpha_train_list = cell(MC_train,1);
    alpha_true_list = cell(MC_train,1);

    for frame = 1:MC_train
       % ===== Noise =====
        noise = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta);
        % ===== Channel =====
        Channel_H = make_channel(cfg, N_r, N_t, var_channel);
        Symbol_X = make_signal(cfg, N_t, L);

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
 
end

 