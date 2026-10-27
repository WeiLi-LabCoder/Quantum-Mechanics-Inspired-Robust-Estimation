function [post_mean] = QD_VAMP(obser, measurement, pri_mean, pri_var, var_noise, x_true, const_Hamil)

[length_row, length_col] = size(measurement);
[U, S, V] = svd(measurement);
y_flag = (1/var_noise) * S' * U' * obser;

Hamil_P_base = Hamiltonian_Original(length_col, 1, const_Hamil);

%% Initialization
gama_2(2) = 1 / pri_var;
x2_mean(:,2) = var_noise * (inv(measurement' * measurement * var_noise + gama_2(2) * eye(length_col))) * measurement' * obser;


x2_alpha = (gama_2(2) / length_col) * trace(inv(measurement' * measurement * var_noise + gama_2(2) * eye(length_col)));
x2_eta = gama_2(2) / x2_alpha; 
gama_1(2) = max(x2_eta - gama_2(2), 1e-6);
r1(:,2) = x2_mean(:,2) * x2_eta / gama_1(2);
 

%% Iteration
T_NUM = 15;
FixedIter = 1;
t_last=[];
for t = 2:T_NUM
    % === Denoising module ===
    if t <= FixedIter
        x1_eta(t) = gama_1(t) + 1 / pri_var;
        x1_mean(:,t) = (r1(:,t) * gama_1(t) + pri_mean / pri_var) / x1_eta(t);
        
        gama_2(t) = x1_eta(t) - gama_1(t);
        r2(:,t) = (x1_mean(:,t) * x1_eta(t) - r1(:,t) * gama_1(t)) / gama_2(t);
    else
        x1_mean(:,t) = DenoisedSignal_QD(Hamil_P_base, r1(:,t), x_true);  

        % Numerical derivative to estimate divergence (eta) 
        epsilon = 1e-3;
        M_mc = 20;  % Monte Carlo 次数
        eta_list = zeros(1, M_mc); 
        for m = 1:M_mc
            temp_z = randn(size(r1(:,t)));
            g_r1_eps = DenoisedSignal_QD(Hamil_P_base, r1(:,t) + epsilon * temp_z, x_true);
            eta_list(m) = (temp_z' * (g_r1_eps - x1_mean(:,t))) / (epsilon * norm(temp_z)^2);
        end
        
        eta(t) = mean(eta_list);  % 平均值作为估计
        eta(t) = max(min(real(eta(t)), 0.95), 1e-3);  % 夹紧在合理范围 

        % Update
        norm_diff = norm(x1_mean(:,t) - r1(:,t))^2;
        if norm_diff < 1e-10 * norm(r1(:,t))^2 || norm_diff < 1e-14
            break;
        end
        gama_2(t) = 1 / (norm_diff / length_col);
        r2(:,t) = (x1_mean(:,t) - eta(t) * r1(:,t)) / (1 - eta(t));
    end

    % === LMMSE module ===
    D = inv((1/var_noise) * S' * S + gama_2(t) * eye(length_col));
    x2_mean(:,t) = V * D * (y_flag + gama_2(t) * V' * r2(:,t));
    
    alpha(t) = sum(diag(D) * gama_2(t)) / length_row;
    r1(:,t+1) = (x2_mean(:,t) - alpha(t) * r2(:,t)) / (1 - alpha(t));
    
    x2_eta_inv(t) = alpha(t);
    gama_1(t+1) = gama_2(t) * (1 - x2_eta_inv(t)) / x2_eta_inv(t);
 

    i_vari(t,1)=0.0;
	a_fact(t,1)=0.0;
    i_vari(t,1)=norm(r1(:,t+1)-r1(:,t))^2;
    a_fact(t,1)=norm(r1(:,t))^2;
    if i_vari(t,1)<a_fact(t,1)*1e-12
        t_last = t;
        break;
    end
    t_last = t;
end

post_mean = x2_mean(:,t_last);
end
