function [Rec_Y_NL_vec,tau,U_P] = DenoisedSignal_QD(Hamil_P_base,Rec_vec_y,Chan_Symbol)
% this function computes the denoised signal through quantum mechanics
v = abs(Rec_vec_y);          
Hamil_P = Hamil_P_base + diag(v);
[U_P, ~, ~] = svd(Hamil_P);

% Mapping to the basis functions
alpha_train =  linsolve(U_P, Rec_vec_y);
alpha_true =  linsolve(U_P, Chan_Symbol(:))  ;

% Trained parameters  

loss_fun = @(b) norm(U_P * (alpha_train .* (1 ./ (1 + exp(b(1)*(abs(alpha_train) - b(2)))))) - U_P * alpha_true)^2;
beta_init = [1; 0.01];
options = optimset('Display','off', 'MaxIter', 300);

beta_old = beta_init;
beta_opt = fminsearch(loss_fun, beta_init, options);

if loss_fun(beta_opt) >= loss_fun(beta_old)
    beta_opt = beta_old;
end


% % optimized the weight
tau = 1 ./ (1 + exp(beta_opt(1)*(abs(alpha_train) - beta_opt(2)))); 
Rec_Y_NL_vec = U_P * (alpha_train .* tau);
end