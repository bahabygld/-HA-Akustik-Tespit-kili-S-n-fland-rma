clear; clc;

rootOut = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V2";
load(fullfile(rootOut,'T_train.mat'),'T_train');

% --- Seçilecek kolonlar: F_* + (LabelCat, File, NoiseType, SNR)
varNames  = T_train.Properties.VariableNames;

featMask  = startsWith(varNames,"F_");
metaNames = {'LabelCat','File','NoiseType','SNR'};

% 1) Maskeyle (kolay yol)
keepMask  = featMask | ismember(varNames, metaNames);
T = T_train(:, keepMask);

% (İstersen sıralı olsun dersen tamamen sayısal indeksle:)
% featIdx = find(featMask);
% metaIdx = find(ismember(varNames, metaNames));
% selIdx  = [featIdx, metaIdx];      % önce F_*, sonra meta
% T = T_train(:, selIdx);

% --- Grup ID (aynı dosyanın tüm varyantları tek grupta)
[~,base,~] = cellfun(@fileparts, cellstr(T.File), 'UniformOutput', false);
gid = string(base);

% --- Grup bazlı 90/10 bölme
u  = unique(gid);
cv = cvpartition(numel(u),'Holdout',0.10);
isValGrp = ismember(gid, u(test(cv)));

idxVAL = isValGrp;
idxTR  = ~isValGrp;

% --- Val-Clean / Val-Noise ayır
isClean = (T.NoiseType=="clean") | isnan(T.SNR);

T_TR        = T(idxTR, :);                  % train (clean + augment hepsi)
T_VAL_clean = T(idxVAL & isClean, :);       % erken durdurma/threshold
T_VAL_noise = T(idxVAL & ~isClean, :);      % robustness raporu

% --- Dışa yaz
writetable(T_TR,         fullfile(rootOut,'TR_features.csv'));
writetable(T_VAL_clean,  fullfile(rootOut,'VAL_clean.csv'));
writetable(T_VAL_noise,  fullfile(rootOut,'VAL_noise.csv'));

disp("✅ Hazır: TR_features.csv, VAL_clean.csv, VAL_noise.csv");

% Hızlı kontrol
fprintf('Train: %d | Val-Clean: %d | Val-Noise: %d\n', ...
    height(T_TR), height(T_VAL_clean), height(T_VAL_noise));
