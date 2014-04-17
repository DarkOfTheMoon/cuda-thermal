typedef struct struct_type 
{
 int N, LX, LY, frame, frame_rate, num_frame, sigma_start,
	 btc_spot, btc_pts, btc_rate, ts, comp,
     	restart_frame_num, solute_start, ic_x1, ic_x2, ic_y1,
	 ic_y2, t, radii, solute_bcs_zerograd_e, solute_bcs_zerograd_w, solute_bcs_zerograd_n, solute_bcs_zerograd_s, solute_bcs_e, solute_bcs_w, solute_bcs_n, solute_bcs_s, pressure_bcs_ew,
 	pressure_bcs_ns, pressure_bcs_tb;
 double tau0, tau1_xx, tau1_yy, gr_x, gr_y, gr_b, t_half,
	ns_ridge, ns_slough, F_gr[Q], beta,
       rho0_in, rho0_out, rho1_in, rho1_bcs, ux_in, uy_in, uz_in,
       sigma, alpha, dRho;
} mystruct;
