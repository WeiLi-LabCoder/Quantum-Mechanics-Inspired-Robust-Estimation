function [Rec_Y_NL_vec,tau,U_P] = DenoisedSignal_QD_trained(Hamil_P_base,Rec_vec_y,beta_opt)
% this function computes the denoised signal through quantum mechanics
v = abs(Rec_vec_y);          
Hamil_P = Hamil_P_base + diag(v);
[U_P, ~, ~] = svd(Hamil_P);

% Mapping to the basis functions
alpha_train =  linsolve(U_P, Rec_vec_y); 

% Trained parameters  
 
% % optimized the weight
tau = 1 ./ (1 + exp(beta_opt(1)*(abs(alpha_train) - beta_opt(2)))); 
Rec_Y_NL_vec = U_P * (alpha_train .* tau);
end