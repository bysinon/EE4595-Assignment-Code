function A = system_matrix(Mr, N, object_locs, u_inc_vec, kb, lambda, pixel_area)
    rec_line_x = [-lambda, 2*lambda];
    rec_line_y = 1.5 * lambda;
    rec_x_vec = linspace(rec_line_x(1), rec_line_x(2), Mr);
    receiver_locs = [rec_x_vec', (ones(Mr, 1) * rec_line_y)];
    A = zeros(Mr, N, 'like', 1j);
    for m_idx = 1:Mr
        for n_idx = 1:N
            rho_m = receiver_locs(m_idx, :);
            rho_n_prime = object_locs(n_idx, :);
            dist = norm(rho_m - rho_n_prime);
            G_mn = (-1j/4) * besselh(0, 2, kb * dist);
            u_inc_n = u_inc_vec(n_idx);
            A(m_idx, n_idx) = G_mn * u_inc_n * pixel_area;
        end
    end
end