%% ------------------- training parameters (beta) ------------------- %% 
function beta_SNR = ParaTrain(SNR_Range,MC_train,cfg,N_r,N_t,L,var_channel,P_sig,delta,Hamil_P_base)

beta_SNR = zeros(2, length(SNR_Range));   

for snr_idx = 1:length(SNR_Range)
    SNR_dB    = SNR_Range(snr_idx);
    var_noise = P_sig / 10^(SNR_dB/10);

    U_P_list = cell(MC_train,1);
    alpha_train_list = cell(MC_train,1);
    alpha_true_list  = cell(MC_train,1);

    for frame = 1:MC_train
        noise = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta);
        [Channel_H, stats] = make_channel(cfg, N_r, N_t, var_channel);
        Symbol_X = qammod(randi([0 cfg.Mconst-1],N_t,L),cfg.Mconst,'UnitAveragePower',true);

        Chan_Symbol = Channel_H * Symbol_X;
        Rec_vec_y   = Chan_Symbol(:) + noise(:);

        Hamil_P = Hamil_P_base + diag(abs(Rec_vec_y));   
        [U_P,~,~] = svd(Hamil_P);

        alpha_train_list{frame} = U_P \ Rec_vec_y;
        alpha_true_list{frame}  = U_P \ Chan_Symbol(:);
        U_P_list{frame} = U_P;
    end

    beta_init = [0.6; 0.01];
    options = optimset('Display','off','MaxIter',500);

    obj = @(b) sum(arrayfun(@(i) ...
        local_loss(U_P_list{i}, alpha_train_list{i}, alpha_true_list{i}, b), ...
        1:MC_train));

    beta_opt = fminsearch(obj, beta_init, options);
    beta_SNR(:, snr_idx) = beta_opt;
end
end

function v = local_loss(U, a_tr, a_true, b)
tau = 1 ./ (1 + exp(b(1) * (abs(a_tr) - b(2))));
res = U*(a_tr .* tau) - U*a_true;
v = norm(res)^2;
end