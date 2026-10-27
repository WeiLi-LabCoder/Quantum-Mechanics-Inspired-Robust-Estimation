% The weight (tau) parameter is optimized in each frame 
% channel and transmitted signal varies in each frame
clear all; clc
% Parameter setting
N_r = 4; N_t = 4; L = 4;
SNR_Range = -20:5:10;
var_channel = 1;
FrameAll = 3000;

hbar = 1; mass = 1;
const_Hamil = hbar^2 / (2*mass);
Hamil_P_base = Hamiltonian_Original(N_r, L, const_Hamil);

err_NL = zeros(1, length(SNR_Range));
err = zeros(1, length(SNR_Range));
beta_store = 1e10*ones(2,2);
MC_train = FrameAll; % training parameters within finite frames
for snr_idx = 1:length(SNR_Range)
    SNR_test = SNR_Range(snr_idx);
    var_noise = 10^(-0.1*SNR_test);
    MC_train_iter = 1;
    for frame = 1:FrameAll 
        % noise
        % noise = sqrt(var_noise) * randn(N_r, L) ; % Gaussian noise 
        noise = poissrnd(var_noise, N_r, L) - var_noise; % Poission noise
        % noise = sqrt(var_noise) * randn(N_r, L) + poissrnd(var_noise, N_r, L) - var_noise; % mixed Gaussian and Poission noise 

        % correlated channel 
        % rho = 0.99; 
        % H_vec = mvnrnd(zeros(N_r*N_t, 1), (1 - rho) * eye(N_r*N_t) + rho * ones(N_r*N_t));   
        % Channel_H = reshape(H_vec, N_r, N_t); 
        Channel_H = sqrt(var_channel) * randn(N_r, N_t);   % i.i.d. channel

        % correlated signal 
        % X_vec = mvnrnd(zeros(N_t*L,1), (1 - rho) * eye(N_t*L) + rho * ones(N_t*L));
        % Symbol_X = reshape(X_vec, N_t, L); 
        % Symbol_X = randn(N_t, L);  % i.i.d. signal
        Symbol_X = randi([0,1],N_t, L); % binary signal


        Chan_Symbol = Channel_H * Symbol_X;
        Rec_Y = Chan_Symbol + noise;
        Rec_vec_y = Rec_Y(:);

        % Construct Hamil_P
        Hamil_P = Hamil_P_base + diag(Rec_vec_y);
        [U_P, ~, ~] = svd(Hamil_P);

        % Mapping to the basis functions
        alpha_train =  linsolve(U_P, Rec_vec_y);
        alpha_true =  linsolve(U_P, Chan_Symbol(:))  ;

        % Trained parameters  
        if frame <= MC_train
            loss_fun = @(b) norm(U_P * (alpha_train .* (1 ./ (1 + exp(b(1)*(abs(alpha_train) - b(2)))))) - U_P * alpha_true)^2;
            beta_init = [1; 0.01];
            options = optimset('Display','off', 'MaxIter', 300);
            beta_opt = fminsearch(loss_fun, beta_init, options);
            beta_store(:,MC_train_iter+1) = beta_opt;
            if loss_fun(beta_opt)<loss_fun(beta_store(:,MC_train_iter))
                beta_opt = beta_opt;
            else
                beta_opt = beta_store(:,MC_train_iter);
            end
        end
 
        % optimized the weight
        tau = 1 ./ (1 + exp(beta_opt(1)*(abs(alpha_train) - beta_opt(2))));
        Rec_Y_NL_vec = U_P * (alpha_train .* tau);

        % compute NMSE in each frame
        err_NL_temp = norm(Rec_Y_NL_vec - Chan_Symbol(:))^2 / norm(Chan_Symbol(:))^2;
        err_temp = norm(Rec_vec_y - Chan_Symbol(:))^2 / norm(Chan_Symbol(:))^2;
         if any(isnan( err_NL_temp)) || any(isinf(err_NL_temp))|| any(isnan( err_temp)) || any(isinf(err_temp))
            err_NL_temp=0; 
            err_temp=0;
         end
        err_NL(snr_idx) = err_NL(snr_idx) + err_NL_temp;
        err(snr_idx) = err(snr_idx) + err_temp; 
    end

    % NMSE in each SNR
    err_NL(snr_idx) = err_NL(snr_idx) / FrameAll;
    err(snr_idx) = err(snr_idx) / FrameAll;
end

% Plot
figure
semilogy(SNR_Range, err_NL, 'or-', 'LineWidth', 1.5)
hold on
semilogy(SNR_Range, err, 'xb-', 'LineWidth', 1.5)
xlabel('SNR (dB)', 'Interpreter', 'latex')
ylabel('Estimation error in $\mathbf{Y}$', 'Interpreter', 'latex')
legend(['Quantum Denoising with optimal $\tau$ ($N_r=$',num2str(N_r),...
    '$, N_t=$',num2str(N_t),'$, L=$',num2str(L),')'], ...
    ['No denoising ($N_r=$',num2str(N_r),'$, N_t=$', ...
    num2str(N_t),'$, L=$',num2str(L),')'], 'Interpreter', 'latex')
grid on
