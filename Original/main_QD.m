clear all
clc
% Parameter setting
N_r = 64;         % receiver ant. 
N_t = 20;         % transmitter ant. 
L = 1;           % length of observation time
SNR_Range = [-10:5:10]; % SNR range
var_channel = 1;

hbar = 20;
mass = 1;
const_Hamil= hbar^2/(2*mass);
% const_Hamil = 2e-7;

Hamil_P=Hamiltonian_Original(N_r,L,const_Hamil);

FrameAll = 100;
iter=1;
err_NL=zeros(1,length(SNR_Range));
err=zeros(1,length(SNR_Range));
for SNR=SNR_Range 
    var_noise = 10^(-0.1*SNR);
    for frame = 1:FrameAll
        noise = sqrt(var_noise)*randn(N_r,L);
        % Input-output relationship
        Channel_H = sqrt(var_channel)*randn(N_r,N_t);
        % Symbol_X = round(rand(N_t,L));
        Symbol_X = rand(N_t,L);
        Chan_Symbol = Channel_H * Symbol_X;
        Rec_Y = Chan_Symbol + noise;
        Rec_vec_y = Rec_Y(:); 
        % Noisy signal-related Hamiltonian matrix
        Hamil_P = Hamil_P + diag(Rec_vec_y);

        [U_P,Sigma_P,V_P]=svd(Hamil_P);   
        alpha = Rec_vec_y.'*U_P;  % projection on basis functions   
        Sigma_P = diag(Sigma_P);
        % thre_num = numel(Sigma_P(Sigma_P(1)./Sigma_P <=3));
        % thre_num_range = rescale(SNR_Range,1,length(Sigma_P));
        thre_num = 3;
        thre_rho = abs(SNR)/10;
        kk = 1:length(Sigma_P);
        weight = max(0, 1 - max(0, kk - thre_num) / thre_rho);
        weight_alpha = weight.*alpha;
        Rec_Y_NL_vec = U_P * weight_alpha.';


       
        % Vs = 2 .^frame;
        % taux = heavi(Rec_vec_y,Vs);
        % for v = Vs
        %     for k = 1:2:N_r*L
        %         taux = heavi(1 - k + (1:N_r*L), v);
        %         n_SB = sum((taux .* U_P) .* U_P.', 2).';
        %     end
        % end

        % [U_P,Sigma_P,V_P]=svd(Hamil_P);   
        % alpha = Rec_vec_y.'*U_P;  % projection on basis functions   
        % Sigma_P = diag(Sigma_P);
        % thre_num = max(1,sum(Sigma_P<(mean(Sigma_P)*SNR)));
        % thre_rho = 2;
        % kk = 1:length(Sigma_P);
        % weight = max(0, 1 - max(0, kk - thre_num) / thre_rho);
        % weight_alpha = alpha.*weight;
        % Rec_Y_NL_vec = U_P * weight_alpha.';


        Rec_Y_NL = reshape(Rec_Y_NL_vec,N_r,L);
         
        % Est_Symbol_X_NL = Channel_H'*inv(Channel_H*Channel_H')*Rec_Y_NL;
        % Est_Symbol_X = Channel_H'*inv(Channel_H*Channel_H')*Rec_Y;
        % err_NL(1,iter)=err_NL(1,iter)+...
        %     norm(Est_Symbol_X_NL-Symbol_X,'fro')^2/(norm(Symbol_X,'fro')^2);  
        % err(1,iter)=err(1,iter)+...
        %     norm(Est_Symbol_X-Symbol_X,'fro')^2/(norm(Symbol_X,'fro')^2);  
        err_NL(1,iter)=err_NL(1,iter)+...
            norm(Rec_Y_NL_vec-Chan_Symbol(:),'fro')^2/(norm(Chan_Symbol(:),'fro')^2);  
        err(1,iter)=err(1,iter)+...
            norm(Rec_vec_y-Chan_Symbol(:),'fro')^2/(norm(Chan_Symbol(:),'fro')^2); 
    end
    err_NL(1,iter) = err_NL(1,iter)/FrameAll;
    err(1,iter) = err(1,iter)/FrameAll;
    iter=iter+1;
end
% err_NL = err_NL/FrameAll;
% err = err/FrameAll;

figure
semilogy(SNR_Range,err_NL,'or-','LineWidth', 1.5)
hold on 
semilogy(SNR_Range,err,'xb-','LineWidth', 1.5)
xlabel('SNR (dB)',Interpreter='latex')
ylabel('estimation error in $\mathbf{Y}$',Interpreter='latex')
legend('Quantum Denoising','No denoising',Interpreter='latex')







