%% =========================================================
% Benchmark: runtime & memory vs problem size
%
% Figures:
%   Fig.1  Online runtime vs problem size
%   Fig.2  End-to-end runtime vs problem size
%   Fig.3  Estimated memory footprint vs problem size
%
% Definitions:
%   fixed           = direct test only
%   trained-online  = trained beta, test only
%   trained-E2E     = ParaTrain + test
%   oracle          = fminsearch + test
%
% Requires existing project functions:
%   Hamiltonian_Original
%   ParaTrain
%   make_noise_alpha_stable
%   make_channel
%   get_mod_order
%   DenoisedSignal_QD_trained
%   DenoisedSignal_QD
%% =========================================================

clear; clc; close all;
rng(1);

%% ================= User settings =================
Nr_list    = [16, 32, 64, 128, 256, 512];
L          = 1;
num_repeat = 10;
use_warmup = true;

SNR_Range  = -5:5:20;
SNR_target = 10;
snr_idx    = find(SNR_Range == SNR_target, 1);
if isempty(snr_idx)
    error('SNR_target must be one element of SNR_Range.');
end

% fixed beta table
beta_SNR_fixed = [0.01, -70, -0.1, -74, -13, -13; ...
                  0.01, 0.03, -300, 0.3, 1.4, 0.8];
beta_fixed = beta_SNR_fixed(:, snr_idx);

% training repetitions inside ParaTrain
MC_train = 1;

fprintf('Benchmark SNR_target = %d dB\n', SNR_target);
fprintf('Fixed beta at SNR=%d dB: [%g; %g]\n', ...
    SNR_target, beta_fixed(1), beta_fixed(2));

%% ================= Common configuration =================
N_t = 8;

var_channel = 5;
cfg.trueVarScale  = 1.0;
cfg.forceMeanBias = 0.0;

cfg.fadingModel     = 'rayleigh';
cfg.K_rician        = 5;
cfg.useCorrChan     = false;
cfg.rho_r           = 0.7;
cfg.rho_t           = 0.5;

cfg.useULAsteering = false;
cfg.d_over_lambda  = 0.5;
cfg.AoA            = [];
cfg.AoD            = [];

cfg.useImperfectCSI = false;
cfg.csiErrVar       = var_channel * 1e-2;

cfg.alpha = 1;

cfg.mod    = '64qam';
cfg.Mconst = get_mod_order(cfg.mod);

cfg.priorCase = 'known';
cfg.partialMode = 'var_mismatch';
cfg.varMismatchFactor = 2.0;
cfg.meanMismatchBias  = 0.3;
cfg.absentMode = 'default01';
cfg.weakPriorVar = 1e3;

P_sig = 1;
delta = 0;
var_noise = P_sig / 10^(SNR_target/10);

hbar = 1;
mass = 1;
const_Hamil = hbar^2 / (2*mass);

%% ================= Collectors =================
Nobs_list = Nr_list * L;

% runtime
runtime_fixed          = zeros(size(Nr_list));
runtime_trained_online = zeros(size(Nr_list));
runtime_trained_e2e    = zeros(size(Nr_list));
runtime_oracle         = zeros(size(Nr_list));

% memory (estimated footprint)
memory_fixed_MB          = zeros(size(Nr_list));
memory_trained_online_MB = zeros(size(Nr_list));
memory_trained_e2e_MB    = zeros(size(Nr_list));
memory_oracle_MB         = zeros(size(Nr_list));

%% ================= Main loop =================
for idx = 1:length(Nr_list)

    N_r  = Nr_list(idx);
    Nobs = N_r * L;

    fprintf('\n====================================================\n');
    fprintf('Benchmarking N_r = %d, L = %d, Nobs = %d\n', N_r, L, Nobs);
    fprintf('====================================================\n');

    Hamil_P_base = Hamiltonian_Original(N_r, L, const_Hamil);
    if ~isequal(size(Hamil_P_base), [Nobs, Nobs])
        error('Hamil_P_base size mismatch: expected [%d %d], got [%d %d].', ...
            Nobs, Nobs, size(Hamil_P_base,1), size(Hamil_P_base,2));
    end

    %% ---------- trained beta for this problem size ----------
    tic;
    beta_SNR_trained = ParaTrain(SNR_target, MC_train, cfg, N_r, N_t, L, ...
                                 var_channel, P_sig, delta, Hamil_P_base);
    t_train_only = toc;
    beta_trained = beta_SNR_trained;

    %% ---------- Fixed: direct test only ----------
    t_fix = zeros(num_repeat,1);
    m_fix = zeros(num_repeat,1);

    for r = 1:num_repeat
        [Rec_vec_y, Chan_Symbol_used] = generate_one_instance( ...
            cfg, N_r, N_t, L, var_channel, var_noise, delta);

        if use_warmup && idx == 1 && r == 1
            DenoisedSignal_QD_trained(Hamil_P_base, Rec_vec_y, beta_fixed);
        end

        tic;
        [Rec_fix, tau_fix, U_fix] = DenoisedSignal_QD_trained( ...
            Hamil_P_base, Rec_vec_y, beta_fixed);
        t_fix(r) = toc;

        m_fix(r) = estimate_memory_fixed_case( ...
            Hamil_P_base, Rec_vec_y, beta_fixed, Rec_fix, tau_fix, U_fix);
    end

    runtime_fixed(idx)   = median(t_fix);
    memory_fixed_MB(idx) = median(m_fix) / (1024^2);

    %% ---------- Trained-online: trained beta, test only ----------
    t_tr_on = zeros(num_repeat,1);
    m_tr_on = zeros(num_repeat,1);

    for r = 1:num_repeat
        [Rec_vec_y, Chan_Symbol_used] = generate_one_instance( ...
            cfg, N_r, N_t, L, var_channel, var_noise, delta);

        if use_warmup && idx == 1 && r == 1
            DenoisedSignal_QD_trained(Hamil_P_base, Rec_vec_y, beta_trained);
        end

        tic;
        [Rec_tr_on, tau_tr_on, U_tr_on] = DenoisedSignal_QD_trained( ...
            Hamil_P_base, Rec_vec_y, beta_trained);
        t_tr_on(r) = toc;

        m_tr_on(r) = estimate_memory_trained_online_case( ...
            Hamil_P_base, Rec_vec_y, beta_trained, Rec_tr_on, tau_tr_on, U_tr_on);
    end

    runtime_trained_online(idx)   = median(t_tr_on);
    memory_trained_online_MB(idx) = median(m_tr_on) / (1024^2);

    %% ---------- Trained-E2E: ParaTrain + test ----------
    % End-to-end = training cost + one representative test cost
    runtime_trained_e2e(idx) = t_train_only + median(t_tr_on);

    m_train_only = estimate_training_memory(beta_SNR_trained);
    memory_trained_e2e_MB(idx) = max([m_train_only, median(m_tr_on)]) / (1024^2);

    %% ---------- Oracle: fminsearch + test ----------
    t_or = zeros(num_repeat,1);
    m_or = zeros(num_repeat,1);

    for r = 1:num_repeat
        [Rec_vec_y, Chan_Symbol_used] = generate_one_instance( ...
            cfg, N_r, N_t, L, var_channel, var_noise, delta);

        if use_warmup && idx == 1 && r == 1
            DenoisedSignal_QD(Hamil_P_base, Rec_vec_y, Chan_Symbol_used);
        end

        tic;
        [Rec_or, tau_or, U_or] = DenoisedSignal_QD( ...
            Hamil_P_base, Rec_vec_y, Chan_Symbol_used);
        t_or(r) = toc;

        m_or(r) = estimate_memory_oracle_case( ...
            Hamil_P_base, Rec_vec_y, Chan_Symbol_used, Rec_or, tau_or, U_or);
    end

    runtime_oracle(idx)   = median(t_or);
    memory_oracle_MB(idx) = median(m_or) / (1024^2);

    fprintf('N=%d | fixed=%.4g s | trained-online=%.4g s | trained-E2E=%.4g s | oracle=%.4g s\n', ...
        N_r, runtime_fixed(idx), runtime_trained_online(idx), ...
        runtime_trained_e2e(idx), runtime_oracle(idx));
end

%% ================= Figure 1: Online runtime =================
figure;
loglog(Nobs_list, runtime_fixed,          '-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
loglog(Nobs_list, runtime_trained_online, '-s', 'LineWidth', 2, 'MarkerSize', 8);
loglog(Nobs_list, runtime_oracle,         '-^', 'LineWidth', 2, 'MarkerSize', 8);
grid on;

xlabel(sprintf('Problem size N = N_r L (here L = %d)', L));
ylabel('Wall-clock runtime (seconds)');
title(sprintf('Online runtime vs problem size (SNR = %d dB)', SNR_target));
legend({'Fixed = direct test', ...
        'Trained-online = trained \beta, test only', ...
        'Oracle = fminsearch + test'}, ...
        'Location', 'northwest');

ax = gca;
ax.XScale = 'log';
ax.XTick = Nobs_list;
ax.XTickLabel = string(Nobs_list);

%% ================= Figure 2: End-to-end runtime =================
figure;
loglog(Nobs_list, runtime_fixed,       '-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
loglog(Nobs_list, runtime_trained_e2e, '-s', 'LineWidth', 2, 'MarkerSize', 8);
loglog(Nobs_list, runtime_oracle,      '-^', 'LineWidth', 2, 'MarkerSize', 8);
grid on;

xlabel(sprintf('Problem size N = N_r L (here L = %d)', L));
ylabel('Wall-clock runtime (seconds)');
title(sprintf('End-to-end runtime vs problem size (SNR = %d dB)', SNR_target));
legend({'Fixed = direct test', ...
        'Trained = ParaTrain + test', ...
        'Oracle = fminsearch + test'}, ...
        'Location', 'northwest');

ax = gca;
ax.XScale = 'log';
ax.XTick = Nobs_list;
ax.XTickLabel = string(Nobs_list);

%% ================= Figure 3: Estimated memory footprint =================
figure;
loglog(Nobs_list, memory_fixed_MB,          '-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
loglog(Nobs_list, memory_trained_online_MB, '-s', 'LineWidth', 2, 'MarkerSize', 8);
loglog(Nobs_list, memory_trained_e2e_MB,    '--s', 'LineWidth', 2, 'MarkerSize', 8);
loglog(Nobs_list, memory_oracle_MB,         '-^', 'LineWidth', 2, 'MarkerSize', 8);
grid on;

xlabel(sprintf('Problem size N = N_r L (here L = %d)', L));
ylabel('Estimated memory footprint (MB)');
title(sprintf('Estimated memory footprint vs problem size (SNR = %d dB)', SNR_target));
legend({'Fixed = direct test', ...
        'Trained-online = trained \beta, test only', ...
        'Trained-E2E = ParaTrain + test', ...
        'Oracle = fminsearch + test'}, ...
        'Location', 'northwest');

ax = gca;
ax.XScale = 'log';
ax.XTick = Nobs_list;
ax.XTickLabel = string(Nobs_list);

%% ================= Tables =================
T_runtime = table(Nr_list(:), repmat(L, numel(Nr_list),1), Nobs_list(:), ...
    runtime_fixed(:), runtime_trained_online(:), runtime_trained_e2e(:), runtime_oracle(:), ...
    'VariableNames', {'N_r','L','Nobs','Runtime_Fixed_s','Runtime_TrainedOnline_s','Runtime_TrainedE2E_s','Runtime_Oracle_s'});

T_memory = table(Nr_list(:), repmat(L, numel(Nr_list),1), Nobs_list(:), ...
    memory_fixed_MB(:), memory_trained_online_MB(:), memory_trained_e2e_MB(:), memory_oracle_MB(:), ...
    'VariableNames', {'N_r','L','Nobs','Memory_Fixed_MB','Memory_TrainedOnline_MB','Memory_TrainedE2E_MB','Memory_Oracle_MB'});

disp('================ Runtime table ================');
disp(T_runtime);

disp('================ Memory table ================');
disp(T_memory);

%% ================= Save =================
results.SNR_target               = SNR_target;
results.Nr_list                  = Nr_list;
results.L                        = L;
results.Nobs_list                = Nobs_list;
results.beta_fixed               = beta_fixed;
results.runtime_fixed            = runtime_fixed;
results.runtime_trained_online   = runtime_trained_online;
results.runtime_trained_e2e      = runtime_trained_e2e;
results.runtime_oracle           = runtime_oracle;
results.memory_fixed_MB          = memory_fixed_MB;
results.memory_trained_online_MB = memory_trained_online_MB;
results.memory_trained_e2e_MB    = memory_trained_e2e_MB;
results.memory_oracle_MB         = memory_oracle_MB;

save('benchmark_runtime_memory_fixed_trained_oracle.mat', ...
     'results', 'T_runtime', 'T_memory');
% exportgraphics(gcf,'fig_RuntimeMemory.pdf','ContentType','vector');
%% =========================================================
% Local helper functions
%% =========================================================

function [Rec_vec_y, Chan_Symbol_used] = generate_one_instance(cfg, N_r, N_t, L, var_channel, var_noise, delta)
    noise = make_noise_alpha_stable(cfg, N_r, L, var_noise, delta);
    [Channel_H, ~] = make_channel(cfg, N_r, N_t, var_channel);

    Symbol_X = qammod(randi([0 cfg.Mconst-1], N_t, L), ...
                      cfg.Mconst, 'UnitAveragePower', true);

    Chan_Symbol_used = Channel_H * Symbol_X;
    Rec_Y = Chan_Symbol_used + noise;
    Rec_vec_y = Rec_Y(:);
end

function bytes = estimate_memory_fixed_case(Hamil_P_base, Rec_vec_y, beta_opt, Rec_Y_NL_vec, tau, U_P)
    v = abs(Rec_vec_y);
    Hamil_P = Hamil_P_base + diag(v);
    alpha_train = U_P' * Rec_vec_y;

    bytes = 0;
    bytes = bytes + var_bytes(Hamil_P_base);
    bytes = bytes + var_bytes(v);
    bytes = bytes + var_bytes(Hamil_P);
    bytes = bytes + var_bytes(U_P);
    bytes = bytes + var_bytes(alpha_train);
    bytes = bytes + var_bytes(tau);
    bytes = bytes + var_bytes(Rec_vec_y);
    bytes = bytes + var_bytes(Rec_Y_NL_vec);
    bytes = bytes + var_bytes(beta_opt);
end

function bytes = estimate_memory_trained_online_case(Hamil_P_base, Rec_vec_y, beta_opt, Rec_Y_NL_vec, tau, U_P)
    % Same online-stage memory type as fixed, except beta comes from training.
    v = abs(Rec_vec_y);
    Hamil_P = Hamil_P_base + diag(v);
    alpha_train = U_P' * Rec_vec_y;

    bytes = 0;
    bytes = bytes + var_bytes(Hamil_P_base);
    bytes = bytes + var_bytes(v);
    bytes = bytes + var_bytes(Hamil_P);
    bytes = bytes + var_bytes(U_P);
    bytes = bytes + var_bytes(alpha_train);
    bytes = bytes + var_bytes(tau);
    bytes = bytes + var_bytes(Rec_vec_y);
    bytes = bytes + var_bytes(Rec_Y_NL_vec);
    bytes = bytes + var_bytes(beta_opt);
end

function bytes = estimate_training_memory(beta_SNR_trained)
    % Minimal storage related to trained beta output.
    bytes = var_bytes(beta_SNR_trained);
end

function bytes = estimate_memory_oracle_case(Hamil_P_base, Rec_vec_y, Chan_Symbol_used, Rec_Y_NL_vec, tau, U_P)
    % Estimated dominant variables for oracle stage.
    % Does NOT include full internal workspace of fminsearch.
    v = abs(Rec_vec_y);
    Hamil_P = Hamil_P_base + diag(v);
    alpha_train = U_P' * Rec_vec_y;
    alpha_true  = U_P' * Chan_Symbol_used(:);

    beta_init = [1; 0.01];
    beta_opt  = beta_init;

    bytes = 0;
    bytes = bytes + var_bytes(Hamil_P_base);
    bytes = bytes + var_bytes(v);
    bytes = bytes + var_bytes(Hamil_P);
    bytes = bytes + var_bytes(U_P);
    bytes = bytes + var_bytes(alpha_train);
    bytes = bytes + var_bytes(alpha_true);
    bytes = bytes + var_bytes(tau);
    bytes = bytes + var_bytes(Rec_vec_y);
    bytes = bytes + var_bytes(Rec_Y_NL_vec);
    bytes = bytes + var_bytes(Chan_Symbol_used);
    bytes = bytes + var_bytes(beta_init);
    bytes = bytes + var_bytes(beta_opt);
end

function b = var_bytes(x)
    s = whos('x');
    b = s.bytes;
end