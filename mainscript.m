clear all;
close all;

%% Solution 3: Sketch of the Configuration
kb = 1;                 % Background wavenumber
lambda = 2 * pi / kb;   % Wavelength

% Object domain D
domain_pos = [0, 0, lambda, lambda];
% Source location
source_loc = [lambda/2, 10*lambda];

figure(2);
hold on;
% object domain D
rectangle('Position', domain_pos, 'EdgeColor', 'b', 'LineWidth', 2, 'LineStyle', '--');
% source location rho_s
plot(source_loc(1), source_loc(2), 'r*', 'MarkerSize', 12, 'LineWidth', 2);
xlabel('x-axis');
ylabel('y-axis');
title('Figure 1: Geometric Configuration');
set(gca, 'YDir', 'reverse','XAxisLocation','top');
grid on;
xlim([-1.5 * lambda, 2.5 * lambda]);
ylim([-1 * lambda, 11 *lambda]);
text(domain_pos(1) + 0.2, domain_pos(2) + lambda/2, 'Object Domain D', 'Color', 'blue', 'FontSize', 10);
text(source_loc(1) + 0.2, source_loc(2), 'Source \rho_s', 'Color', 'red', 'FontSize', 10);

%% Solution 5: uniform grid
M = 20; % Number of pixels per dimension
h = lambda / M;
x_vec = linspace(0, lambda, M);
y_vec = linspace(0, lambda, M);
[X, Y] = meshgrid(x_vec, y_vec);
N = numel(X); % Total number of pixels

figure(2);
line(X, Y);
line(Y, X);
text(lambda + 0.2, lambda, '(\lambda,\lambda)')
set(gca, 'YDir', 'reverse', 'XAxisLocation', 'top');
title('Figure 2: Uniform grid, with h = \lambda/20');

%% Q6  Incident field
% distance from the source to every grid point
rho = sqrt((X - source_loc(1)).^2 + (Y - source_loc(2)).^2);
% Use the Green's function formula
u_inc = (-1i/4) * besselh(0, 2, kb * rho);

figure();
subplot(1,3,1);
imagesc(x_vec, y_vec, real(u_inc)); colorbar;
title('Real Part of u^{inc}'); xlabel('x-axis'); ylabel('y-axis');
set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight;
subplot(1,3,2);
imagesc(x_vec, y_vec, imag(u_inc)); colorbar;
title('Imaginary Part of u^{inc}'); xlabel('x-axis'); ylabel('y-axis');
set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight;
subplot(1,3,3);
imagesc(x_vec, y_vec, abs(u_inc)); colorbar;
title('Absolute Value of u^{inc}'); xlabel('x-axis'); ylabel('y-axis');
set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight;

%% Q7
% for a closer source
source_loc_closer = [lambda/2, 2*lambda];
rho_closer = sqrt((X - source_loc_closer(1)).^2 + (Y - source_loc_closer(2)).^2);
u_inc_closer = (-1i/4) * besselh(0, 2, kb * rho_closer);
figure();
subplot(1,3,1); imagesc(x_vec, y_vec, real(u_inc_closer)); colorbar; title('Real Part of u^{inc}'); set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight;
subplot(1,3,2); imagesc(x_vec, y_vec, imag(u_inc_closer)); colorbar; title('Imaginary Part of u^{inc}'); set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight;
subplot(1,3,3); imagesc(x_vec, y_vec, abs(u_inc_closer)); colorbar; title('Absolute Value of u^{inc}'); set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight;

% for set kb = 2
kb_new = 2;
rho_kbchanged = sqrt((X - source_loc(1)).^2 + (Y - source_loc(2)).^2);
u_inc_kbchanged = (-1i/4) * besselh(0, 2, kb_new * rho_kbchanged);
figure();
subplot(1,3,1); imagesc(x_vec, y_vec, real(u_inc_kbchanged)); colorbar; title('Real Part of u^{inc}'); set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight;
subplot(1,3,2); imagesc(x_vec, y_vec, imag(u_inc_kbchanged)); colorbar; title('Imaginary Part of u^{inc}'); set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight;
subplot(1,3,3); imagesc(x_vec, y_vec, abs(u_inc_kbchanged)); colorbar; title('Absolute Value of u^{inc}'); set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight;

%% Q8
% first circle
contrast = zeros(M, M);
center1 = [lambda/2, lambda/3];
radius1 = lambda / 6;
contrast_value = 1.0;
circle1 = (X - center1(1)).^2 + (Y - center1(2)).^2 <= radius1^2;
contrast(circle1) = contrast_value;
% second circle
center2 = [lambda/2, lambda/1.5];
radius2 = lambda / 5;
circle2 = (X - center2(1)).^2 + (Y - center2(2)).^2 <= radius2^2;
contrast(circle2) = contrast_value;

%% Q9
figure('Name', 'Object Contrast Function');
imagesc(x_vec, y_vec, contrast);
colorbar; colormap('parula');
set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight;

%% Q10 & Q11
rec_line_x = [-lambda, 2*lambda];
rec_line_y = 1.5 * lambda;
figure(2);
line(rec_line_x, [rec_line_y, rec_line_y], 'LineWidth', 1.5); hold on;
scatter(rec_line_x, [rec_line_y, rec_line_y], 'filled');
text(-lambda, 1.5 * lambda + 3, '(-\lambda,1.5\lambda)', 'Color', 'red', 'FontSize', 10);
text(2 * lambda, 1.5 * lambda + 3, '(2\lambda,1.5\lambda)', 'Color', 'red', 'FontSize', 10);

Mr = 200;
rec_x_vec = linspace(rec_line_x(1), rec_line_x(2), Mr);
receiver_locs = [rec_x_vec', (ones(Mr, 1) * rec_line_y)];
figure(2);
scatter(rec_x_vec, ones(1, Mr) * rec_line_y, 10, 'g', 'filled');

%% Q13
x_true = reshape(contrast, N, 1);

%% Q14
X_vec = reshape(X, N, 1);
Y_vec = reshape(Y, N, 1);
object_locs = [X_vec, Y_vec];
pixel_area = h^2;
u_inc_vec = reshape(u_inc, N, 1);
A = system_matrix(Mr, N, object_locs, u_inc_vec, kb, lambda, pixel_area);

%% Q15
figure('Name', 'Singular Value Decay');
s_vals = svd(A);
plot(log10(s_vals / max(s_vals)));
grid on;
title('Singular Values of System Matrix A');
xlabel('Singular Value Index');
ylabel('Normalized Singular Values (log)');

%% Q16
d = A * x_true;

%% Q17
x_mn = pinv(A) * d;

%% Q18
contrast_reconstructed = reshape(x_mn, M, M);
figure('Name', 'Comparison of Original and Reconstructed Contrast');
subplot(1, 2, 1);
imagesc(x_vec, y_vec, contrast);
set(gca, 'YDir', 'reverse','XAxisLocation','top');
axis equal tight; colorbar;
title('Original Contrast \chi'); xlabel('x-axis'); ylabel('y-axis');
subplot(1, 2, 2);
imagesc(x_vec, y_vec, abs(contrast_reconstructed));
set(gca, 'YDir', 'reverse','XAxisLocation','top');
axis equal tight; colorbar;
title('Reconstructed Contrast |x_{mn}|'); xlabel('x-axis');

%% Q19
reconstruct_contrast_image(M, N, object_locs, u_inc_vec, x_true, kb, lambda, pixel_area, 20, 'Reconstruction with Fewer Receivers (Mr=20)');
reconstruct_contrast_image(M, N, object_locs, u_inc_vec, x_true, kb, lambda, pixel_area, 100, 'Reconstruction with Fewer Receivers (Mr=100)');
reconstruct_contrast_image(M, N, object_locs, u_inc_vec, x_true, kb, lambda, pixel_area, 400, 'Reconstruction with Fewer Receivers (Mr=400)');

%% Q20
Mr = 200;
A_baseline = system_matrix(Mr, N, object_locs, u_inc_vec, kb, lambda, pixel_area);
d_noise_free = A_baseline * x_true;
noise_level = 0.05;
noise_vec = (randn(Mr, 1) + 1i * randn(Mr, 1));
noise = noise_vec / norm(noise_vec) * norm(d_noise_free) * noise_level;
d_noisy = d_noise_free + noise;
x_noisy_unreg = pinv(A_baseline) * d_noisy;
contrast_noisy_unreg = reshape(x_noisy_unreg, M, M);

figure('Name', 'Influence of Noise without Regularization');
subplot(1, 2, 1);
imagesc(x_vec, y_vec, contrast); set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight; colorbar;
title('Original Contrast'); xlabel('x'); ylabel('y');
subplot(1, 2, 2);
imagesc(x_vec, y_vec, abs(contrast_noisy_unreg)); set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight; colorbar;
title('Reconstruction from Noisy Data'); xlabel('x');

%% Q21
s_vals = svd(A_baseline);
s_squared = s_vals.^2;
total_energy = sum(s_squared);
cumulative_energy = cumsum(s_squared);
r0 = find(cumulative_energy >= 0.999 * total_energy, 1, 'first');
[U, S, V] = svd(A_baseline);
U_r0 = U(:, 1:r0);
S_r0 = S(1:r0, 1:r0);
V_r0 = V(:, 1:r0);
x_tsvd = V_r0 * (S_r0 \ (U_r0' * d_noisy));
contrast_tsvd = reshape(x_tsvd, M, M);

% Born iteration
num_iterations = 3;
x_k = zeros(N, 1);
u_k = u_inc_vec;
I_N = eye(N);
s_vals_A = svd(system_matrix(Mr, N, object_locs, u_inc_vec, kb, lambda, pixel_area));
tolerance = s_vals_A(1) * 1e-2;
G_D = zeros(N, N, 'like', 1j);
for i = 1:N
    for j = 1:N
        if i == j; continue; end
        dist = norm(object_locs(i, :) - object_locs(j, :));
        G_D(i, j) = (-1j/4) * besselh(0, 2, kb * dist) * pixel_area;
    end
end

for k = 1:num_iterations
    fprintf('Born Iteration: %d/%d\n', k, num_iterations);
    A_k = system_matrix(Mr, N, object_locs, u_k, kb, lambda, pixel_area);
    d_residual = d_noisy - A_k * x_k;
    delta_x = pinv(A_k, tolerance) * d_residual;
    x_k = x_k + delta_x;
    fprintf('Updating total field inside the object...\n');
    u_k = (I_N - G_D * spdiags(x_k, 0, N, N)) \ u_inc_vec;
end

contrast_born_iterative = reshape(x_k, M, M);

figure('Name', 'Final Comparison of Methods');
subplot(1,2,1);
imagesc(x_vec, y_vec, abs(contrast_tsvd)); set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight; colorbar;
title('TSVD Result');
subplot(1,2,2);
imagesc(x_vec, y_vec, abs(contrast_born_iterative)); set(gca, 'YDir', 'reverse','XAxisLocation','top'); axis equal tight; colorbar;
title('Born Iterative Result');