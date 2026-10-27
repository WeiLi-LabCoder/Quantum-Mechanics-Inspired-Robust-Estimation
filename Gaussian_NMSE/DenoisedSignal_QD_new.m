function varargout = DenoisedSignal_QD_new(Hamil_P_base, y_vec, ref_vec)
% DenoisedSignal_QD
% Usage:
%   [y_tilde, tau, U_P] = DenoisedSignal_QD(Hamil_P_base, y_vec, ref_vec)  % supervised (optional)
%   y_tilde             = DenoisedSignal_QD(Hamil_P_base, y_vec)           % unsupervised / PnP
%   y_tilde             = DenoisedSignal_QD(Hamil_P_base, y_vec, [])        % unsupervised / PnP
%
% Notes:
%   - This function NO LONGER requires ref_vec.
%   - If ref_vec is provided (non-empty), we optionally compute a "best tau" by minimizing MSE
%     in the spectral domain (still lightweight), but without solving with linsolve(U_P, ref).
%     We use alpha_ref = U_P' * ref_vec (since U_P is orthonormal).
%   - If ref_vec is empty, tau is chosen adaptively based on noise scale estimated from y_vec.

    if nargin < 3
        ref_vec = [];
    end

    y_vec = y_vec(:);

    % If Hamil_P_base is already the Hamiltonian matrix P:
    % Many implementations pass a base Laplacian and then add diag(y). If your original code
    % does something else, adapt here. We'll assume Hamil_P_base is a matrix to which diag(y) is added.
    % P = Hamil_P_base + diag(y_vec);  % typical
    %
    % If your original Hamiltonian construction differs, keep your original construction here,
    % but ensure it DOES NOT use ref_vec.
    P = Hamil_P_base + diag(y_vec);

    % Eigen-decomposition (P is symmetric/Hermitian real)
    % Use eig for full basis; if large, consider eigs.
    [U_P, Lambda] = eig((P + P')/2);
    % Sort by eigenvalue ascending (optional but consistent)
    [lam, idx] = sort(real(diag(Lambda)), 'ascend');
    U_P = U_P(:, idx);

    % Spectral coefficients of y
    alpha_y = U_P' * y_vec;

    if ~isempty(ref_vec)
        % -------- Supervised tau (still doesn't use linsolve) --------
        ref_vec = ref_vec(:);
        if numel(ref_vec) ~= numel(y_vec)
            error('DenoisedSignal_QD: ref_vec length mismatch.');
        end
        alpha_ref = U_P' * ref_vec;

        % A simple "oracle" shrinkage in spectral domain:
        % tau_i = clip( (alpha_ref_i / alpha_y_i), 0, 1 ) with guard.
        % This mimics ideal attenuation when y is noisy.
        eps0 = 1e-12;
        tau = abs(alpha_ref) ./ max(abs(alpha_y), eps0);
        tau = min(max(tau, 0), 1);

    else
        % -------- Unsupervised tau for PnP --------
        % Robust noise scale from y spectrum tail:
        % estimate sigma via MAD on high-frequency coefficients
        n = numel(alpha_y);
        k0 = max(round(0.7*n), 1); % use top 30% as "mostly noise"
        tail = alpha_y(k0:end);
        sigma_hat = median(abs(tail - median(tail))) / 0.6745;
        sigma_hat = max(sigma_hat, 1e-6);

        % Sigmoid shrinkage: tau_i = 1 / (1 + exp(beta1*(|a|-beta2)))
        beta1 = 2.0 / sigma_hat;      % slope scales with noise
        beta2 = 2.5 * sigma_hat;      % threshold scales with noise

        tau = 1 ./ (1 + exp(beta1 * (abs(alpha_y) - beta2)));
        tau = min(max(real(tau), 0), 1);
    end

    % Reconstruct denoised
    alpha_tilde = tau .* alpha_y;
    y_tilde = U_P * alpha_tilde;

    % Outputs: flexible
    if nargout <= 1
        varargout{1} = y_tilde;
    elseif nargout == 2
        varargout{1} = y_tilde;
        varargout{2} = tau;
    else
        varargout{1} = y_tilde;
        varargout{2} = tau;
        varargout{3} = U_P;
    end
end