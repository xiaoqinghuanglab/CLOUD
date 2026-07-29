// Model 2a: Longitudinal IRT with SDE Latent Process (Joint Latent Sampler - Lyapunov Refactor)

data {
    int<lower=1> N;
    int<lower=1> Nsub;

    int<lower=1> K;
    int<lower=1> R;
    int<lower=1> p;

    array[N] int<lower=1, upper=Nsub> ID;

    array[Nsub] int cumu;
    array[Nsub] int repme;

    array[N, K] int Y;
    array[N, K] int missing_ID;

    vector[N] deltat;

    matrix[N, p] X;

    int<lower=2> ncate4;
    int<lower=2> ncate5;
    int<lower=2> ncate6;
    int<lower=2> ncate7;
}

parameters {

    real theta1; 
    real theta2; 
    real theta3;

    ordered[ncate4 - 1] theta4;
    ordered[ncate5 - 1] theta5;
    ordered[ncate6 - 1] theta6;
    ordered[ncate7 - 1] theta7;

    real mu_theta;
    real<lower=1e-6> sigma_theta;

    vector<lower=1e-6>[K] lambda;
    real<lower=1e-6> sigma_lambda;

    matrix[K, p] beta;

    matrix[Nsub, K] b_raw;
    vector<lower=1e-6>[K] sigma_bk;

    // --- NON-CENTERED JOINT SAMPLER: Raw Noise ---
    matrix[R, N] xi_raw;

    cholesky_factor_cov[R] L_S;   // SPD component
    real gamma_skew;              // skew symmetric strength

    // Replaced real rho with Cholesky factor of the correlation matrix for Omega
    cholesky_factor_corr[R] L_Omega_corr; 
}

transformed parameters {

    matrix[R,R] S;
    matrix[R,R] A;
    matrix[R,R] Gamma;

    matrix[Nsub, K] b;

    matrix[R, R] Omega;

    // --- NON-CENTERED JOINT SAMPLER: Target Matrix ---
    matrix[R, N] xi;

    // SPD component
    S = L_S * L_S';

    // skew symmetric component
    A = rep_matrix(0, R, R);
    A[1,2] = gamma_skew;
    A[2,1] = -gamma_skew;

    // Construct Omega directly as a correlation matrix using the Cholesky factor
    Omega = multiply_lower_tri_self_transpose(L_Omega_corr);

    // Compute strictly identified Drift via the Lyapunov equation solution
    Gamma = mdivide_right(S + A, Omega);

    for (i in 1 : Nsub){
        for (k in 1 : K){
            b[i, k] = b_raw[i, k] * sigma_bk[k];
        }
    }

    // --- NON-CENTERED JOINT SAMPLER: Deterministic Construction ---
    {
        // Cholesky factor of the stationary covariance is explicitly our parameter
        matrix[R, R] L_Omega = L_Omega_corr;
        
        for (i in 1:Nsub) {
            int start_idx = cumu[i] - repme[i] + 1;
            
            // Time = 1
            xi[:, start_idx] = L_Omega * xi_raw[:, start_idx];

            // Time = 2 to end
            for (j in 2:repme[i]) {
                int k = start_idx + j - 1;

                matrix[R, R] Phi = matrix_exp(-deltat[k] * Gamma);
                
                // Stable Covariance Matrix Calculation
                matrix[R, R] Q = Omega - Phi * Omega * Phi';
                matrix[R, R] Q_stable = 0.5 * (Q + Q'); // Ensure symmetry
                matrix[R, R] L_Q = cholesky_decompose(add_diag(Q_stable, 1e-6));
                
                // xi[k] = mean + Standard_Deviation * Noise
                xi[:, k] = Phi * xi[:, k-1] + L_Q * xi_raw[:, k];
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

    mu_theta ~ normal(0, 10);
    sigma_theta ~ cauchy(0, 5);

    lambda ~ normal(1, sigma_lambda);
    sigma_lambda ~ cauchy(0, 5);

    to_vector(beta) ~ cauchy(0, 5);

    sigma_bk ~ cauchy(0, 5);
    to_vector(b_raw) ~ normal(0, 1);

    to_vector(L_S) ~ normal(0,2);
    gamma_skew ~ normal(0,2);

    // Prior for Omega Cholesky
    L_Omega_corr ~ lkj_corr_cholesky(2.0);

    // --- NON-CENTERED JOINT SAMPLER: Isotropic Prior ---
    to_vector(xi_raw) ~ std_normal();

    // Likelihood

    for (i in 1:N) {
        int sub = ID[i];
        row_vector[p] Xi_row = X[i, ];
        
        // Factor 1 items (1-3)
        // Note: matrix indices updated to xi[factor_idx, time_idx]
        if (missing_ID[i, 1] == 0) Y[i, 1] ~ bernoulli_logit(theta1 + Xi_row * beta[1, ]' + lambda[1] * xi[1, i] + b[sub, 1]);
        if (missing_ID[i, 2] == 0) Y[i, 2] ~ bernoulli_logit(theta2 + Xi_row * beta[2, ]' + lambda[2] * xi[1, i] + b[sub, 2]);
        if (missing_ID[i, 3] == 0) Y[i, 3] ~ bernoulli_logit(theta3 + Xi_row * beta[3, ]' + lambda[3] * xi[1, i] + b[sub, 3]);

        // Factor 2 items (4-7)
        if (missing_ID[i, 4] == 0) Y[i, 4] ~ ordered_logistic(Xi_row * beta[4, ]' + lambda[4] * xi[2, i] + b[sub, 4], theta4);
        if (missing_ID[i, 5] == 0) Y[i, 5] ~ ordered_logistic(Xi_row * beta[5, ]' + lambda[5] * xi[2, i] + b[sub, 5], theta5);
        if (missing_ID[i, 6] == 0) Y[i, 6] ~ ordered_logistic(Xi_row * beta[6, ]' + lambda[6] * xi[2, i] + b[sub, 6], theta6);
        if (missing_ID[i, 7] == 0) Y[i, 7] ~ ordered_logistic(Xi_row * beta[7, ]' + lambda[7] * xi[2, i] + b[sub, 7], theta7);
    }
}

generated quantities {
    // --- 1. Structural Matrices Reporting ---
    matrix[R, R] Sigma;
    matrix[R, R] L_Sigma;
    real rho;

    // The diffusion matrix is strictly twice the symmetric component
    Sigma = 2 * S;

    // Calculate identified L_Sigma just for reporting
    L_Sigma = cholesky_decompose(add_diag(Sigma, 1e-6));

    // Recover the explicit correlation parameter
    rho = Omega[1, 2];

    // Stability checks for different time intervals
    matrix[R, R] A05 = matrix_exp(-0.5 * Gamma);
    matrix[R, R] Cova_trans05 = Omega - A05 * Omega * A05';

    matrix[R, R] A10 = matrix_exp(-Gamma);
    matrix[R, R] Cova_trans10 = Omega - A10 * Omega * A10';

    matrix[R, R] A15 = matrix_exp(-1.5 * Gamma);
    matrix[R, R] Cova_trans15 = Omega - A15 * Omega * A15';
/*
    // --- 2. Posterior Predictive Checks & LOO Metrics ---
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

            // Factor 2 (Items 4-7: Ordered Logistic)
            if (missing_ID[i, 4] == 0) {
                real eta = Xi_row * beta[4, ]' + lambda[4] * xi[2, i] + b[sub, 4];
                Y_rep[i, 4] = ordered_logistic_rng(eta, theta4);
                log_lik[i] += ordered_logistic_lpmf(Y[i, 4] | eta, theta4);
                sum_Y[4] += Y[i, 4]; sum_Y_rep[4] += Y_rep[i, 4];
                sum_total += Y[i, 4]; sum_total_rep += Y_rep[i, 4];
            }
           if (missing_ID[i, 5] == 0) {
                real eta = Xi_row * beta[5, ]' + lambda[5] * xi[2, i] + b[sub, 5];
                Y_rep[i, 5] = ordered_logistic_rng(eta, theta5);
                log_lik[i] += ordered_logistic_lpmf(Y[i, 5] | eta, theta5);
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
            if (missing_ID[i, 7] == 0) {
                real eta = Xi_row * beta[7, ]' + lambda[7] * xi[2, i] + b[sub, 7];
                Y_rep[i, 7] = ordered_logistic_rng(eta, theta7);
                log_lik[i] += ordered_logistic_lpmf(Y[i, 7] | eta, theta7);
                sum_Y[7] += Y[i, 7]; sum_Y_rep[7] += Y_rep[i, 7];
                sum_total += Y[i, 7]; sum_total_rep += Y_rep[i, 7];
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