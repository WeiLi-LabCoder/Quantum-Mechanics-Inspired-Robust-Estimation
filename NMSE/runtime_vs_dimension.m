%% =========================================================
%  Benchmark: QD Denoiser Runtime & Memory vs Problem Size
%  (Deployable version, no oracle, no VAMP)
% =========================================================

clear; clc; close all;
rng(1);   % 固定随机种子（保证可复现）

% ---------------- Benchmark sizes ----------------
N_list = [256, 512, 1024, 2048];   % problem size N
num_repeat = 10;                  % repeat for median runtime

% ---------------- Fixed QD parameters ----------------
K = 20;                % number of eigenvectors (fixed)
beta1 = 5;             % fixed shrinkage slope
beta2 = 0.1;           % fixed shrinkage threshold

% Hamiltonian physical constants
hbar = 1;
mass = 1;
const_Hamil = hbar^2/(2*mass);

runtime_sec = zeros(size(N_list));
memory_MB   = zeros(size(N_list));

for idx = 1:length(N_list)

    N = N_list(idx);
    fprintf('Running benchmark for N = %d\n', N);

    % ----- Synthetic observation (no oracle) -----
    y = randn(N,1);

    % ----- Build sparse Hamiltonian base -----
    H_base = Hamiltonian_1D_sparse(N, const_Hamil);

    % ----- Runtime measurement -----
    t = zeros(num_repeat,1);
    for r = 1:num_repeat
        tic;
        QD_denoise_fixed(y, H_base, K, beta1, beta2);
        t(r) = toc;
    end
    runtime_sec(idx) = median(t);

    % ----- Memory estimation -----
    % Sparse Hamiltonian + K eigenvectors + vectors
    nnzH = nnz(H_base);
    bytes = ...
        nnzH * 8 + ...        % sparse values
        nnzH * 8 + ...        % sparse indices (approx)
        N*K * 8 + ...         % eigenvectors
        3*N * 8;              % y, alpha, tau

    memory_MB(idx) = bytes / (1024^2);
end

%% ================= Plot results =================
figure;
loglog(N_list, runtime_sec, '-o', 'LineWidth', 2);
grid on;
xlabel('Problem size N');
ylabel('Runtime (seconds)');
title('QD denoiser runtime vs problem size');

figure;
loglog(N_list, memory_MB, '-s', 'LineWidth', 2);
grid on;
xlabel('Problem size N');
ylabel('Memory usage (MB)');
title('QD denoiser memory vs problem size');

disp(table(N_list(:), runtime_sec(:), memory_MB(:), ...
    'VariableNames', {'N','Runtime_sec','Memory_MB'}));


function H = Hamiltonian_1D_sparse(N, c)
% 1D discrete Laplacian with Neumann boundary

e = ones(N,1);
H = spdiags([-c*e 2*c*e -c*e], [-1 0 1], N, N);

% Neumann boundaries
H(1,1)   = c;
H(N,N)   = c;
end

function y_hat = QD_denoise_fixed(y, H_base, K, beta1, beta2)
% Deployable QD denoiser (no oracle, no training)

N = length(y);

% Construct Hamiltonian (sparse)
H = H_base + spdiags(abs(y), 0, N, N);

% Partial eigensolver
opts.isreal = true;
opts.issym  = true;
[U,~] = eigs(H, K, 'smallestreal', opts);

% Spectral shrinkage
alpha = U' * y;
tau   = 1 ./ (1 + exp(beta1*(abs(alpha)-beta2)));

% Reconstruction
y_hat = U * (alpha .* tau);
end