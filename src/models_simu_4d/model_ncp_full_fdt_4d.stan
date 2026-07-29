data {
    int<lower=1> N;
    int<lower=1> Nsub;
    int<lower=1> K;                   // Number of items (12)
    int<lower=1> R;                   // Number of latent dimensions (4)
    int<lower=1> p;                   // Item covariates
    int<lower=1> q;                   // Number of latent-level covariates (including time)

    array[N] int<lower=1, upper=Nsub> ID;
    array[Nsub] int cumu;
    array[Nsub] int repme;
    array[N, K] int Y;
    array[N, K] int missing_ID;

    vector[N] deltat;
    vector[N] time;                   // Absolute time for mean function

    matrix[N, p] X;                   // Item covariates
    matrix[N, q] Z;                   // Latent mean covariates (e.g., age, gender, etc.)

    // Number of categories for ordinal items (Items 6 through 12)
    int<lower=2> ncate6;
    int<lower=2> ncate7;
    int<lower=2> ncate8;
    int<lower=2> ncate9;
    int<lower=2> ncate10;
    int<lower=2> ncate11;
    int<lower=2> ncate12;
}

parameters {
    // Thresholds for Binary Items (1-5)
    real theta1; 
    real theta2; 
    real theta3;
    real theta4; 
    real theta5;

    // Thresholds for Ordinal Items (6-12)
    ordered[ncate6 - 1] theta6;
    ordered[ncate7 - 1] theta7;
    ordered[ncate8 - 1] theta8;
    ordered[ncate9 - 1] theta9;
    ordered[ncate10 - 1] theta10;
    ordered[ncate11 - 1] theta11;
    ordered[ncate12 - 1] theta12;

    real mu_theta;
    real<lower=1e-6> sigma_theta;

    vector<lower=1e-6>[K] lambda;
    real<lower=1e-6> sigma_lambda;

    matrix[K, p] beta;

    matrix[R, q] A_latent;            // slope on covariates * time
    vector[R] c_latent;               // global intercept

    matrix[Nsub, K] b_raw;
    vector<lower=1e-6>[K] sigma_bk;

    // --- NON-CENTERED JOINT SAMPLER: Raw Noise ---
    matrix[R, N] xi_raw;

    cholesky_factor_cov[R] L_S;       // SPD component
    
    // 6 elements for the strictly upper triangle of a 4x4 matrix
    vector[6] gamma_skew;
    
    // Replaced L_Sigma_corr with direct correlation matrix for Omega
    cholesky_factor_corr[R] L_Omega_corr;
}

transformed parameters {
    matrix[R,R] S;
    matrix[R,R] A;
    matrix[R,R] Gamma;
    matrix[Nsub, K] b;
    
    // Structural Matrices
    matrix[R, R] Omega;

    // --- NON-CENTERED JOINT SAMPLER: Target Matrix ---
    matrix[R, N] xi;
    
    // SPD component
    S = L_S * L_S';
    
    // Skew symmetric component mapped from the 6 parameters
    A = rep_matrix(0, R, R);
    A[1, 2] =  gamma_skew[1]; A[2, 1] = -gamma_skew[1];
    A[1, 3] =  gamma_skew[2]; A[3, 1] = -gamma_skew[2];
    A[1, 4] =  gamma_skew[3]; A[4, 1] = -gamma_skew[3];
    A[2, 3] =  gamma_skew[4]; A[3, 2] = -gamma_skew[4];
    A[2, 4] =  gamma_skew[5]; A[4, 2] = -gamma_skew[5];
    A[3, 4] =  gamma_skew[6]; A[4, 3] = -gamma_skew[6];
    
    // Construct Omega directly as a correlation matrix (Scale = 1 for IRT)
    Omega = multiply_lower_tri_self_transpose(L_Omega_corr);

    // Compute strictly identified Drift
    Gamma = mdivide_right(S + A, Omega);

    for (i in 1 : Nsub){
        for (k in 1 : K){
            b[i, k] = b_raw[i, k] * sigma_bk[k];
        }
    }

    // --- NON-CENTERED JOINT SAMPLER: Deterministic Construction ---
    {
        for (i in 1:Nsub) {
            int start_idx = cumu[i] - repme[i] + 1;
            
            // Time = 1
            vector[q] z0 = Z[start_idx]';
            vector[R] mu_start = (A_latent * z0 + c_latent) * time[start_idx];
            
            // L_Omega_corr is exactly the Cholesky factor of Omega
            xi[:, start_idx] = mu_start + L_Omega_corr * xi_raw[:, start_idx];
            
            // Time = 2 to end
            for (j in 2:repme[i]) {
                int k = start_idx + j - 1;
                matrix[R, R] Phi = matrix_exp(-deltat[k] * Gamma);
                
                // Stable Covariance Matrix Calculation
                matrix[R, R] Q = Omega - Phi * Omega * Phi';
                matrix[R, R] Q_stable = 0.5 * (Q + Q'); // Ensure symmetry
                matrix[R, R] L_Q = cholesky_decompose(add_diag(Q_stable, 1e-6));
                
                vector[q] zk = Z[k]';
                vector[q] zk_prev = Z[k-1]';

                vector[R] target_k = (A_latent * zk + c_latent) * time[k];
                vector[R] target_prev = (A_latent * zk_prev + c_latent) * time[k-1];
                
                vector[R] cond_mean = target_k + Phi * (xi[:, k-1] - target_prev);
                
                // xi[k] = mean + Standard_Deviation * Noise
                xi[:, k] = cond_mean + L_Q * xi_raw[:, k];
            }
        }
    }
}

model {
    // Priors
    theta1 ~ normal(mu_theta, sigma_theta);
    theta2 ~ normal(mu_theta, sigma_theta);
    theta3 ~ normal(mu_theta, sigma_theta);
    theta4 ~ normal(mu_theta, sigma_theta);
    theta5 ~ normal(mu_theta, sigma_theta);

    theta6 ~ normal(mu_theta, sigma_theta);
    theta7 ~ normal(mu_theta, sigma_theta);
    theta8 ~ normal(mu_theta, sigma_theta);
    theta9 ~ normal(mu_theta, sigma_theta);
    theta10 ~ normal(mu_theta, sigma_theta);
    theta11 ~ normal(mu_theta, sigma_theta);
    theta12 ~ normal(mu_theta, sigma_theta);

    mu_theta ~ normal(0, 5);      
    sigma_theta ~ normal(0, 2);

    lambda ~ normal(1, sigma_lambda);
    sigma_lambda ~ normal(0, 2);

    to_vector(beta) ~ normal(0, 5); 

    to_vector(A_latent) ~ normal(0, 2);
    c_latent ~ normal(0, 0.5);

    sigma_bk ~ normal(0, 1);
    to_vector(b_raw) ~ normal(0, 1);

    to_vector(L_S) ~ normal(0, 2);
    gamma_skew ~ normal(0, 2);
    
    // Prior for Omega Cholesky
    L_Omega_corr ~ lkj_corr_cholesky(2.0);
    
    // --- NON-CENTERED JOINT SAMPLER: Isotropic Prior ---
    to_vector(xi_raw) ~ std_normal();
    
    // Likelihood
    for (i in 1:N) {
        int sub = ID[i];
        row_vector[p] Xi_row = X[i, ];
        
        // Factor 1 items (1-3) -- All Binary
        if (missing_ID[i, 1] == 0) Y[i, 1] ~ bernoulli_logit(theta1 + Xi_row * beta[1, ]' + lambda[1] * xi[1, i] + b[sub, 1]);
        if (missing_ID[i, 2] == 0) Y[i, 2] ~ bernoulli_logit(theta2 + Xi_row * beta[2, ]' + lambda[2] * xi[1, i] + b[sub, 2]);
        if (missing_ID[i, 3] == 0) Y[i, 3] ~ bernoulli_logit(theta3 + Xi_row * beta[3, ]' + lambda[3] * xi[1, i] + b[sub, 3]);
        
        // Factor 2 items (4-6) -- Items 4,5 Binary; Item 6 Ordinal
        if (missing_ID[i, 4] == 0) Y[i, 4] ~ bernoulli_logit(theta4 + Xi_row * beta[4, ]' + lambda[4] * xi[2, i] + b[sub, 4]);
        if (missing_ID[i, 5] == 0) Y[i, 5] ~ bernoulli_logit(theta5 + Xi_row * beta[5, ]' + lambda[5] * xi[2, i] + b[sub, 5]);
        if (missing_ID[i, 6] == 0) Y[i, 6] ~ ordered_logistic(Xi_row * beta[6, ]' + lambda[6] * xi[2, i] + b[sub, 6], theta6);
        
        // Factor 3 items (7-9) -- All Ordinal
        if (missing_ID[i, 7] == 0) Y[i, 7] ~ ordered_logistic(Xi_row * beta[7, ]' + lambda[7] * xi[3, i] + b[sub, 7], theta7);
        if (missing_ID[i, 8] == 0) Y[i, 8] ~ ordered_logistic(Xi_row * beta[8, ]' + lambda[8] * xi[3, i] + b[sub, 8], theta8);
        if (missing_ID[i, 9] == 0) Y[i, 9] ~ ordered_logistic(Xi_row * beta[9, ]' + lambda[9] * xi[3, i] + b[sub, 9], theta9);
        
        // Factor 4 items (10-12) -- All Ordinal
        if (missing_ID[i, 10] == 0) Y[i, 10] ~ ordered_logistic(Xi_row * beta[10, ]' + lambda[10] * xi[4, i] + b[sub, 10], theta10);
        if (missing_ID[i, 11] == 0) Y[i, 11] ~ ordered_logistic(Xi_row * beta[11, ]' + lambda[11] * xi[4, i] + b[sub, 11], theta11);
        if (missing_ID[i, 12] == 0) Y[i, 12] ~ ordered_logistic(Xi_row * beta[12, ]' + lambda[12] * xi[4, i] + b[sub, 12], theta12);
    }
}

generated quantities {
    // --- 1. Structural Matrices Reporting ---
    matrix[R, R] Sigma;
    matrix[R, R] L_Sigma;
    vector[6] rho;
    
    // The diffusion matrix is strictly twice the symmetric component
    Sigma = 2 * S;
    
    // Safely decompose Sigma for downstream Python evaluation scripts
    L_Sigma = cholesky_decompose(add_diag(Sigma, 1e-6));
    
    // Extract the 6 upper-triangular correlations directly for Python evaluation scripts
    rho[1] = Omega[1, 2];
    rho[2] = Omega[1, 3];
    rho[3] = Omega[1, 4];
    rho[4] = Omega[2, 3];
    rho[5] = Omega[2, 4];
    rho[6] = Omega[3, 4];

    // Stability checks for different time intervals
    matrix[R, R] A05 = matrix_exp(-0.5 * Gamma);
    matrix[R, R] Cova_trans05 = Omega - A05 * Omega * A05';

    matrix[R, R] A10 = matrix_exp(-Gamma);
    matrix[R, R] Cova_trans10 = Omega - A10 * Omega * A10';

    matrix[R, R] A15 = matrix_exp(-1.5 * Gamma);
    matrix[R, R] Cova_trans15 = Omega - A15 * Omega * A15';
/*
    // --- 2. Posterior Predictive Checks ---
    array[N, K] int Y_rep; 
    vector[N] log_lik;     
    vector[K] ppp_item;
    real ppp_total;

    // Scoped local block to avoid polluting global environment
    {
        vector[K] sum_Y = rep_vector(0.0, K);
        vector[K] sum_Y_rep = rep_vector(0.0, K);
        real sum_total = 0.0;
        real sum_total_rep = 0.0;

        for (i in 1:N) {
            int sub = ID[i];
            row_vector[p] Xi_row = X[i, ];
            
            log_lik[i] = 0.0; // Initialize log-likelihood for visit i
            
            // Assign dummy values for Y_rep default (helps spot unchecked missing values)
            for(k in 1:K) {
                Y_rep[i, k] = -999;
            }

            // Factor 1 (Items 1-3: Bernoulli)
            if (missing_ID[i, 1] == 0) {
                real eta = theta1 + Xi_row * beta[1, ]' + lambda[1] * xi[1, i] + b[sub, 1];
                Y_rep[i, 1] = bernoulli_logit_rng(eta);
                log_lik[i] += bernoulli_logit_lpmf(Y[i, 1] | eta);
                sum_Y[1] += Y[i, 1]; sum_Y_rep[1] += Y_rep[i, 1];
                sum_total += Y[i, 1]; sum_total_rep += Y_rep[i, 1];
            }
            if (missing_ID[i, 2] == 0) {
                real eta = theta2 + Xi_row * beta[2, ]' + lambda[2] * xi[1, i] + b[sub, 2];
                Y_rep[i, 2] = bernoulli_logit_rng(eta);
                log_lik[i] += bernoulli_logit_lpmf(Y[i, 2] | eta);
                sum_Y[2] += Y[i, 2]; sum_Y_rep[2] += Y_rep[i, 2];
                sum_total += Y[i, 2]; sum_total_rep += Y_rep[i, 2];
            }
            if (missing_ID[i, 3] == 0) {
                real eta = theta3 + Xi_row * beta[3, ]' + lambda[3] * xi[1, i] + b[sub, 3];
                Y_rep[i, 3] = bernoulli_logit_rng(eta);
                log_lik[i] += bernoulli_logit_lpmf(Y[i, 3] | eta);
                sum_Y[3] += Y[i, 3]; sum_Y_rep[3] += Y_rep[i, 3];
                sum_total += Y[i, 3]; sum_total_rep += Y_rep[i, 3];
            }

            // Factor 2 (Items 4-5: Bernoulli, Item 6: Ordinal)
            if (missing_ID[i, 4] == 0) {
                real eta = theta4 + Xi_row * beta[4, ]' + lambda[4] * xi[2, i] + b[sub, 4];
                Y_rep[i, 4] = bernoulli_logit_rng(eta);
                log_lik[i] += bernoulli_logit_lpmf(Y[i, 4] | eta);
                sum_Y[4] += Y[i, 4]; sum_Y_rep[4] += Y_rep[i, 4];
                sum_total += Y[i, 4]; sum_total_rep += Y_rep[i, 4];
            }
            if (missing_ID[i, 5] == 0) {
                real eta = theta5 + Xi_row * beta[5, ]' + lambda[5] * xi[2, i] + b[sub, 5];
                Y_rep[i, 5] = bernoulli_logit_rng(eta);
                log_lik[i] += bernoulli_logit_lpmf(Y[i, 5] | eta);
                sum_Y[5] += Y[i, 5]; sum_Y_rep[5] += Y_rep[i, 5];
                sum_total += Y[i, 5]; sum_total_rep += Y_rep[i, 5];
            }
            if (missing_ID[i, 6] == 0) {
                real eta = Xi_row * beta[6, ]' + lambda[6] * xi[2, i] + b[sub, 6];
                Y_rep[i, 6] = ordered_logistic_rng(eta, theta6);
                log_lik[i] += ordered_logistic_lpmf(Y[i, 6] | eta, theta6);
                sum_Y[6] += Y[i, 6]; sum_Y_rep[6] += Y_rep[i, 6];
                sum_total += Y[i, 6]; sum_total_rep += Y_rep[i, 6];
            }

            // Factor 3 (Items 7-9: Ordinal)
            if (missing_ID[i, 7] == 0) {
                real eta = Xi_row * beta[7, ]' + lambda[7] * xi[3, i] + b[sub, 7];
                Y_rep[i, 7] = ordered_logistic_rng(eta, theta7);
                log_lik[i] += ordered_logistic_lpmf(Y[i, 7] | eta, theta7);
                sum_Y[7] += Y[i, 7]; sum_Y_rep[7] += Y_rep[i, 7];
                sum_total += Y[i, 7]; sum_total_rep += Y_rep[i, 7];
            }
            if (missing_ID[i, 8] == 0) {
                real eta = Xi_row * beta[8, ]' + lambda[8] * xi[3, i] + b[sub, 8];
                Y_rep[i, 8] = ordered_logistic_rng(eta, theta8);
                log_lik[i] += ordered_logistic_lpmf(Y[i, 8] | eta, theta8);
                sum_Y[8] += Y[i, 8]; sum_Y_rep[8] += Y_rep[i, 8];
                sum_total += Y[i, 8]; sum_total_rep += Y_rep[i, 8];
            }
            if (missing_ID[i, 9] == 0) {
                real eta = Xi_row * beta[9, ]' + lambda[9] * xi[3, i] + b[sub, 9];
                Y_rep[i, 9] = ordered_logistic_rng(eta, theta9);
                log_lik[i] += ordered_logistic_lpmf(Y[i, 9] | eta, theta9);
                sum_Y[9] += Y[i, 9]; sum_Y_rep[9] += Y_rep[i, 9];
                sum_total += Y[i, 9]; sum_total_rep += Y_rep[i, 9];
            }

            // Factor 4 (Items 10-12: Ordinal)
            if (missing_ID[i, 10] == 0) {
                real eta = Xi_row * beta[10, ]' + lambda[10] * xi[4, i] + b[sub, 10];
                Y_rep[i, 10] = ordered_logistic_rng(eta, theta10);
                log_lik[i] += ordered_logistic_lpmf(Y[i, 10] | eta, theta10);
                sum_Y[10] += Y[i, 10]; sum_Y_rep[10] += Y_rep[i, 10];
                sum_total += Y[i, 10]; sum_total_rep += Y_rep[i, 10];
            }
            if (missing_ID[i, 11] == 0) {
                real eta = Xi_row * beta[11, ]' + lambda[11] * xi[4, i] + b[sub, 11];
                Y_rep[i, 11] = ordered_logistic_rng(eta, theta11);
                log_lik[i] += ordered_logistic_lpmf(Y[i, 11] | eta, theta11);
                sum_Y[11] += Y[i, 11]; sum_Y_rep[11] += Y_rep[i, 11];
                sum_total += Y[i, 11]; sum_total_rep += Y_rep[i, 11];
            }
            if (missing_ID[i, 12] == 0) {
                real eta = Xi_row * beta[12, ]' + lambda[12] * xi[4, i] + b[sub, 12];
                Y_rep[i, 12] = ordered_logistic_rng(eta, theta12);
                log_lik[i] += ordered_logistic_lpmf(Y[i, 12] | eta, theta12);
                sum_Y[12] += Y[i, 12]; sum_Y_rep[12] += Y_rep[i, 12];
                sum_total += Y[i, 12]; sum_total_rep += Y_rep[i, 12];
            }
        }

        // Calculate item-level and total PPP values
        for (k in 1:K) {
            ppp_item[k] = (sum_Y_rep[k] >= sum_Y[k]) ? 1.0 : 0.0;
        }
        ppp_total = (sum_total_rep >= sum_total) ? 1.0 : 0.0;
    }
*/
}