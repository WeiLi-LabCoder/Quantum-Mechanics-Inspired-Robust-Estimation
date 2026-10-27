function Rec_Y_NL_vec=SignalDenoise(Rec_vec_y,Chan_Symbol,Hamil_P_base,controlPara)
% denoise the noisy signal
% controlPara=0 means no interaction term
 % Compute interaction term  
        InterTerm_temp = Rec_vec_y * Rec_vec_y';  % (N_r*L) x (N_r*L) 
        % scaling_factors = diag(InterTerm_temp);  
        % InterTerm_temp = InterTerm_temp ./ scaling_factors.';  
        InterTerm_temp(1:size(InterTerm_temp,1)+1:end) = 0;
        InterTerm = sum(abs(InterTerm_temp),1); 
        % Construct Hamil_P
        Hamil_P = Hamil_P_base + diag(Rec_vec_y) + controlPara*diag(InterTerm); 
        [U_P, ~, ~] = svd(Hamil_P);

        % Mapping to the basis functions
        alpha_train =  linsolve(U_P, Rec_vec_y);
        alpha_true =  linsolve(U_P, Chan_Symbol(:))  ;

        % Trained parameters   
        loss_fun = @(b) norm(U_P * (alpha_train .* (1 ./ (1 + exp(b(1)*(abs(alpha_train) - b(2)))))) - U_P * alpha_true)^2;
        beta_init = [1; 0.01];
        options = optimset('Display','off', 'MaxIter', 300);
        beta_opt = fminsearch(loss_fun, beta_init, options); 
 
        % optimized the weight
        tau = 1 ./ (1 + exp(beta_opt(1)*(abs(alpha_train) - beta_opt(2))));
        Rec_Y_NL_vec = U_P * (alpha_train .* tau);

end