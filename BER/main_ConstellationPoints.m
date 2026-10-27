%% ================= Constellation Comparison =================
clear; clc;  rng(1);

 
cfg.alpha = 2;           
cfg.mod   = '4qam';
cfg.Mconst = get_mod_order(cfg.mod);
cfg.isComplex = 1;

N = 5000;                
L = 1;
SNR_dB = 10;

P_sig = 1;
delta = 0;

 
var_noise = P_sig / 10^(SNR_dB/10);
 
Symbol_X = qammod(randi([0 cfg.Mconst-1], N, L), ...
                  cfg.Mconst, 'UnitAveragePower', true);

 
noise = make_noise_alpha_stable(cfg, N, L, var_noise, delta);

 
Rec_Y = Symbol_X + noise;
 
const_ref = qammod(0:cfg.Mconst-1, cfg.Mconst, 'UnitAveragePower', true);

 
figure('Color','w','Position',[200 200 1100 450]);

 
subplot(1,2,1)
plot(real(Symbol_X), imag(Symbol_X), 'b.', 'MarkerSize', 8); hold on
plot(real(const_ref), imag(const_ref),'ro','MarkerFaceColor','r','MarkerSize',8)

xline(0,'g--','LineWidth',1.5)
yline(0,'g--','LineWidth',1.5)

grid on
axis equal

xlabel('In-phase Component')
ylabel('Quadrature Component')

title('Ideal 16QAM Constellation')

lim = max(abs([real(Symbol_X); imag(Symbol_X)]));
xlim([-1.2*lim 1.2*lim])
ylim([-1.2*lim 1.2*lim])

 
subplot(1,2,2)
plot(real(Rec_Y), imag(Rec_Y), 'b.', 'MarkerSize', 8); hold on
plot(real(const_ref), imag(const_ref),'ro','MarkerFaceColor','r','MarkerSize',8)

xline(0,'g--','LineWidth',1.5)
yline(0,'g--','LineWidth',1.5)

grid on
axis equal

xlabel('In-phase Component')
ylabel('Quadrature Component')

title(sprintf('Noisy 16QAM Constellation (SNR = %d dB, alpha = %g)', SNR_dB, cfg.alpha))

lim = max(abs([real(Rec_Y); imag(Rec_Y)]));
xlim([-1.2*lim 1.2*lim])
ylim([-1.2*lim 1.2*lim])