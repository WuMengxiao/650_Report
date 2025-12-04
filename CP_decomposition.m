
clear; clc; close all;

%% parameters (could be adjusted)
R            = 30; % CP rank
M_ensemble   = 10; % ensemble 
p_keep       = 0.8;
day_to_plot  = 31;

%% data
data  = load('MODIS_Aug.mat');
train = double(data.training_tensor);
test  = double(data.test_tensor);

MaskTrain_full = train > 0;
MaskTest = test  > 0;

[n1,n2,n3] = size(train);

DaysObserved = sum(MaskTrain_full, 3);
figure;
imagesc(DaysObserved);
axis image; colorbar;
title('Number of Observed Days per Pixel (Training)');
xlabel('Longitude index'); ylabel('Latitude index');

%% Bootstrap ensemble
Xhat_all = zeros(n1,n2,n3,M_ensemble);

for m = 1:M_ensemble
    fprintf('Bootstrap ensemble %d / %d\n', m, M_ensemble);

    keep_mask = (rand(n1,n2,n3) < p_keep) & MaskTrain_full;

    MaskTen_m = tensor(keep_mask);
    Xmasked_m = tensor(train .* keep_mask);   % 缺失位置为 0

    % initial: random
    init_cp = 'rand';

    [M_cp, ~] = cp_wopt(Xmasked_m, MaskTen_m, R, 'init', init_cp);

    XhatTen = full(M_cp);
    Xhat_all(:,:,:,m) = double(XhatTen);
end

%% Predictive mean & std
Xhat_mean = mean(Xhat_all, 4);
Xhat_std = std(Xhat_all,  0, 4);

%% metrics
idxTest = (MaskTest == 1);

y_true = test(idxTest);
y_pred = Xhat_mean(idxTest);
y_std  = Xhat_std(idxTest);

rmse = sqrt(mean((y_pred - y_true).^2));
SS_res = sum((y_true - y_pred).^2);
SS_tot = sum((y_true - mean(y_true)).^2);
R2 = 1 - SS_res / SS_tot;

crps_vals = gaussian_crps(y_pred, y_std, y_true);
CRPS = mean(crps_vals);

fprintf('\n=== Bootstrap CP Evaluation on test set ===\n');
fprintf('CP rank %d, ensemble %d, keep %.0f%%\n', R, M_ensemble, p_keep*100);
fprintf('RMSE  = %.3f K\n', rmse);
fprintf('R^2   = %.3f\n', R2);
fprintf('CRPS  = %.3f K\n', CRPS);

%% Visualization: mean (full) vs test
k = day_to_plot;

Test_day = squeeze(test(:,:,k));
Pred_day = squeeze(Xhat_mean(:,:,k));

valid_vals = [Test_day(Test_day>0); Pred_day(:)];
vmin = min(valid_vals);
vmax = max(valid_vals);

figure;
subplot(1,2,1);
imagesc(Test_day);
axis image; colorbar; caxis([vmin, vmax]);
title(sprintf('Test Ground Truth (Day %d)', k));
xlabel('Longitude index'); ylabel('Latitude index');

subplot(1,2,2);
imagesc(Pred_day);
axis image; colorbar; caxis([vmin, vmax]);
title(sprintf('Bootstrap CP Mean Prediction (Day %d)', k));
xlabel('Longitude index'); ylabel('Latitude index');
sgtitle('Test vs Predicted Mean (Kelvin)');

%% Visualization: std (full)
Std_day = squeeze(Xhat_std(:,:,k));

figure;
imagesc(Std_day);
axis image; colorbar;
title(sprintf('Predictive Std (Day %d) - Full domain', k));
xlabel('Longitude index'); ylabel('Latitude index');

%% Visualization: only missing (mean & std)
MissingTrain_day = ~MaskTrain_full(:,:,k);

Mean_missing = Pred_day;
Mean_missing(~MissingTrain_day) = NaN;   % 把非 missing 设成 NaN

Std_missing = Std_day;
Std_missing(~MissingTrain_day) = NaN;

figure;
subplot(1,2,1);
imagesc(Mean_missing);
axis image; colorbar;
title(sprintf('Mean Prediction on Missing Region (Day %d)', k));
xlabel('Longitude index'); ylabel('Latitude index');

subplot(1,2,2);
imagesc(Std_missing);
axis image; colorbar;
title(sprintf('Predictive Std on Missing Region (Day %d)', k));
xlabel('Longitude index'); ylabel('Latitude index');

sgtitle('Only Missing Region: Mean & Std');

%% CRPS for Gaussian function
function crps = gaussian_crps(mu, sigma, y)
    sigma = max(sigma, 1e-6);
    z   = (y - mu) ./ sigma;
    Phi = 0.5 * (1 + erf(z / sqrt(2)));
    phi = (1 / sqrt(2*pi)) * exp(-0.5 * z.^2);
    crps = sigma .* ( z .* (2.*Phi - 1) + 2.*phi - 1/sqrt(pi) );
end
