function reconstruct_contrast_image(M, N, object_locs, u_inc_vec, x_true, kb, lambda, pixel_area, Mr, fig_title)
    A = system_matrix(Mr, N, object_locs, u_inc_vec, kb, lambda, pixel_area);
    d = A * x_true;
    x_mn = pinv(A) * d;
    contrast_reconstructed = reshape(x_mn, M, M);
    contrast_true_2d = reshape(x_true, M, M);
    x_vec = linspace(0, lambda, M);
    y_vec = linspace(0, lambda, M);

    figure('Name', fig_title);
    subplot(1, 2, 1);
    imagesc(x_vec, y_vec, contrast_true_2d);
    set(gca, 'YDir', 'reverse','XAxisLocation','top');
    axis equal tight; colorbar;
    title('Original Contrast'); xlabel('x'); ylabel('y');
    
    subplot(1, 2, 2);
    imagesc(x_vec, y_vec, abs(contrast_reconstructed));
    set(gca, 'YDir', 'reverse','XAxisLocation','top');
    axis equal tight; colorbar;
    title(sprintf('Reconstruction (Mr=%d)', Mr)); xlabel('x');
end