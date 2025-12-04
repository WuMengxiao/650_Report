clear; clc; close all;

%% parameters

% Could be adjusted
ranks_list = [  6  8  3;
                8 10  3;
               10 12  4;
               30 50 10];

% Cross-validation settings
cv_frac      = 0.10;   % fraction of observed train entries held out for CV
nIterEM_cv   = 5;      % EM iterations for CV
maxiters_cv  = 40;     % max iters in each tucker_als during CV

% Ensemble (probabilistic) settings
M_ensemble   = 50;     % ensemble size
p_keep       = 0.8;    % bootstrap keep probability
nIterEM_ens  = 5;      % EM iterations for ensemble
maxiters_ens = 40;     % max iters in each tucker_als during ensemble

% Visualization: which day (1–31) to inspect
day_to_plot  = 31;     

rng(123);  % for reproducibility (random seeds)

%% data

data  = load('MODIS_Aug.mat');
train = double(data.training_tensor);
test  = double(data.test_tensor);

[n1,n2,n3] = size(train);

MaskTrain_full = (train ~= 0);
MaskTest       = (test  ~= 0);

fprintf('Data loaded: %d x %d x %d (lat x lon x day)\n', n1, n2, n3);

%% rank selection (by cross-validation)

cv_mask = false(size(train));
r = rand(n1,n2,n3);
cv_mask(MaskTrain_full) = r(MaskTrain_full) < cv_frac;
MaskTrain_cv = MaskTrain_full & ~cv_mask;

nRanks       = size(ranks_list,1);
RMSE_cv      = zeros(nRanks,1);
RMSE_test    = zeros(nRanks,1);
R2_test      = zeros(nRanks,1);
Xhat_det_all = cell(nRanks,1);   % store deterministic reconstructions

for r_idx = 1:nRanks
    ranks = ranks_list(r_idx,:);
    fprintf('\n CV for Tucker rank [%d %d %d] \n', ...
            ranks(1), ranks(2), ranks(3));

    % initial filling
    X_obs    = train .* MaskTrain_cv;
    vals0    = X_obs(MaskTrain_cv);
    mean0    = mean(vals0);              % mean of observed entries
    X_filled = X_obs;
    X_filled(~MaskTrain_cv) = mean0;     % initial guess for missing

    % EM
    for it = 1:nIterEM_cv
        T = tensor(X_filled);
        M = tucker_als(T, ranks, ...
                       'init','random', ...
                       'maxiters', maxiters_cv, ...
                       'tol', 1e-5, ...
                       'printitn', 0);
        X_recon = double(full(M));
        X_filled(~MaskTrain_cv) = X_recon(~MaskTrain_cv);
    end

    Xhat_rank = X_recon; 
    Xhat_det_all{r_idx} = Xhat_rank;

    % RMSE for CV
    diff_cv = Xhat_rank(cv_mask) - train(cv_mask);
    RMSE_cv(r_idx) = sqrt(mean(diff_cv.^2));
    fprintf('CV RMSE = %.3f K\n', RMSE_cv(r_idx));

    % RMSE & R^2
    idxTest = (MaskTest == 1);
    y_true  = test(idxTest);
    y_pred  = Xhat_rank(idxTest);

    RMSE_test(r_idx) = sqrt(mean((y_pred - y_true).^2));
    SS_res           = sum((y_true - y_pred).^2);
    SS_tot           = sum((y_true - mean(y_true)).^2);
    R2_test(r_idx)   = 1 - SS_res/SS_tot;

    fprintf('Test RMSE = %.3f K, Test R^2 = %.3f\n', ...
            RMSE_test(r_idx), R2_test(r_idx));
end

% Choose best rank by smallest CV RMSE
[~, best_idx] = min(RMSE_cv);
best_rank = ranks_list(best_idx,:);

fprintf('\n Selected Tucker rank (by CV) = [%d %d %d] \n', ...
        best_rank(1), best_rank(2), best_rank(3));
fprintf('CV RMSE = %.3f K, Test RMSE = %.3f K, Test R^2 = %.3f\n', ...
        RMSE_cv(best_idx), RMSE_test(best_idx), R2_test(best_idx));

% Deterministic reconstruction with best rank (for CV error map)
Xhat_det = Xhat_det_all{best_idx};

%% Probabilistic Tucker (bootstrap ensemble)

ranks = best_rank;
fprintf('\n Probabilistic Tucker (rank [%d %d %d]) \n', ...
        ranks(1), ranks(2), ranks(3));

Xhat_all = zeros(n1,n2,n3,M_ensemble);

for m = 1:M_ensemble
    fprintf('  Ensemble %d / %d\n', m, M_ensemble);

    keep_mask = (rand(n1,n2,n3) < p_keep) & MaskTrain_full;
    X_obs_m   = train .* keep_mask;
    vals_m    = X_obs_m(keep_mask);
    if isempty(vals_m)
        vals_m = train(MaskTrain_full);
    end
    mean_m = mean(vals_m);

    % initial fill
    X_filled = X_obs_m;
    X_filled(~keep_mask) = mean_m;

    % EM
    for it = 1:nIterEM_ens
        T = tensor(X_filled);
        M = tucker_als(T, ranks, ...
                       'init','random', ...
                       'maxiters', maxiters_ens, ...
                       'tol', 1e-5, ...
                       'printitn', 0);
        X_recon = double(full(M));
        X_filled(~keep_mask) = X_recon(~keep_mask);
    end

    Xhat_all(:,:,:,m) = X_recon;
end

% Ensemble mean & std
Xhat_mean = mean(Xhat_all,4); % predictive mean
Xhat_std  = std(Xhat_all,0,4); % predictive std

%% metrics, CRPS

idxTest = (MaskTest == 1);
y_true  = test(idxTest);
y_pred  = Xhat_mean(idxTest);
y_sd    = Xhat_std(idxTest);

rmse_ens = sqrt(mean((y_pred - y_true).^2));
SS_res   = sum((y_true - y_pred).^2);
SS_tot   = sum((y_true - mean(y_true)).^2);
R2_ens   = 1 - SS_res/SS_tot;

crps_vec = gaussian_crps(y_pred, y_sd, y_true);
CRPS_ens = mean(crps_vec);

fprintf('\n Probabilistic Tucker metrics \n');
fprintf('RMSE  = %.3f K\n', rmse_ens);
fprintf('R^2   = %.3f\n',  R2_ens);
fprintf('CRPS  = %.3f K\n', CRPS_ens);

% put pointwise CRPS back to 3D tensor
CRPS_tensor = nan(size(test));
CRPS_tensor(idxTest) = crps_vec;

%% mean, std, and CRPS (full domain)

k = day_to_plot;

Test_day = squeeze(test(:,:,k));
Mean_day = squeeze(Xhat_mean(:,:,k));
Std_day  = squeeze(Xhat_std(:,:,k));
CRPS_day = squeeze(CRPS_tensor(:,:,k));

Test_day_plot = Test_day;
Test_day_plot(~MaskTest(:,:,k)) = NaN;

valid_vals = [Test_day(Test_day>0); Mean_day(:)];
vmin = min(valid_vals);
vmax = max(valid_vals);

figure;
subplot(1,3,1);
imagesc(Test_day_plot);
axis image; colorbar; caxis([vmin,vmax]);
title(sprintf('Test GT (Day %d)',k));
xlabel('Lon idx'); ylabel('Lat idx');

subplot(1,3,2);
imagesc(Mean_day);
axis image; colorbar; caxis([vmin,vmax]);
title(sprintf('Tucker mean (Day %d)',k));
xlabel('Lon idx'); ylabel('Lat idx');

subplot(1,3,3);
imagesc(Std_day);
axis image; colorbar;
title(sprintf('Tucker std (Day %d)',k));
xlabel('Lon idx'); ylabel('Lat idx');

sgtitle(sprintf('Full domain – rank [%d %d %d]', ...
        ranks(1),ranks(2),ranks(3)));

figure;
imagesc(CRPS_day);
axis image; colorbar;
title(sprintf('CRPS on test pixels (Day %d)',k));
xlabel('Lon idx'); ylabel('Lat idx');

%% plot missing only

% training missing on this day
MissingTrain_day = ~MaskTrain_full(:,:,k);   
Test_mask_day    = MaskTest(:,:,k);

GT_missing   = Test_day_plot;
GT_missing(~Test_mask_day) = NaN;

Mean_missing = Mean_day;
Mean_missing(~Test_mask_day) = NaN;

figure;
subplot(1,2,1);
imagesc(GT_missing);
axis image; colorbar;
title(sprintf('GT (missing pixels, Day %d)',k));
xlabel('Lon idx'); ylabel('Lat idx');

subplot(1,2,2);
imagesc(Mean_missing);
axis image; colorbar;
title(sprintf('Reconstructed mean (missing, Day %d)',k));
xlabel('Lon idx'); ylabel('Lat idx');

sgtitle('GT vs reconstruction – missing region only');

% 7.2 mean / std / CRPS on missing region
Std_missing  = Std_day;
Std_missing(~Test_mask_day) = NaN;

CRPS_missing = CRPS_day;
CRPS_missing(~Test_mask_day) = NaN;

figure;
subplot(1,3,1);
imagesc(Mean_missing);
axis image; colorbar;
title(sprintf('Mean (missing, Day %d)',k));
xlabel('Lon idx'); ylabel('Lat idx');

subplot(1,3,2);
imagesc(Std_missing);
axis image; colorbar;
title(sprintf('Std (missing, Day %d)',k));
xlabel('Lon idx'); ylabel('Lat idx');

subplot(1,3,3);
imagesc(CRPS_missing);
axis image; colorbar;
title(sprintf('CRPS (missing, Day %d)',k));
xlabel('Lon idx'); ylabel('Lat idx');

sgtitle('Missing-only: mean / std / CRPS');

%% CV error map

cv_err = nan(size(train));
cv_err(cv_mask) = Xhat_det(cv_mask) - train(cv_mask);

AbsErr = abs(cv_err);
Err_day  = squeeze(cv_err(:,:,k));
AE_day   = squeeze(AbsErr(:,:,k));

figure;
subplot(1,2,1);
imagesc(Err_day);
axis image; colorbar;
title(sprintf('CV error (recon - true), Day %d',k));
xlabel('Lon idx'); ylabel('Lat idx');

subplot(1,2,2);
imagesc(AE_day);
axis image; colorbar;
title(sprintf('CV |error|, Day %d',k));
xlabel('Lon idx'); ylabel('Lat idx');

sgtitle('Cross-validation error maps (held-out train)');

%% Gaussian CRPS function
function crps = gaussian_crps(mu, sigma, y)
    % Gaussian CRPS at each point (mu, sigma, y)
    sigma = max(sigma, 1e-6);     % avoid zero std
    z   = (y - mu) ./ sigma;
    Phi = 0.5 * (1 + erf(z ./ sqrt(2)));
    phi = (1 ./ sqrt(2*pi)) .* exp(-0.5 .* z.^2);
    crps = sigma .* ( z .* (2.*Phi - 1) + 2.*phi - 1./sqrt(pi) );
end
