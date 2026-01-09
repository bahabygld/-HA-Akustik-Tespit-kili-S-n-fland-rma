% ================================================
% group_based_eval.m
% DADS - Group-based 5-Fold CV (File/Source-aware)
% Kıyas: Fine KNN (k=1) ve Gaussian SVM (RBF)
% ================================================
clc; clear; close all;

% --- 1. Veri ---
matFile = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\İha Akustik ses\Veri_Setleri\ozellik_cikarimi\dads_features_light.mat";
S = load(matFile);
T = S.T_clip_light;

% --- 2. Kontroller ---
if ~iscategorical(T.label), T.label = categorical(T.label); end
assert(ismember('source_id', T.Properties.VariableNames), 'source_id bulunamadı.');
feats = startsWith(T.Properties.VariableNames,'F_');
X = T{:,feats};
y = T.label;
G = string(T.source_id);

K = 5;
cv = cvpartition(G,'KFold',K);

% --- 3. Değerlendirme ---
models = {'knn','svm'};
for m = 1:numel(models)
    modelName = models{m};
    balAcc = zeros(K,1);
    macroF1 = zeros(K,1);

    fprintf('\n=== MODEL: %s ===\n', upper(modelName));
    for k = 1:K
        tr = training(cv,k); te = test(cv,k);

        mu = mean(X(tr,:)); sig = std(X(tr,:));
        Xz = (X - mu) ./ max(sig,1e-8);

        switch modelName
            case 'knn'
                mdl = fitcknn(Xz(tr,:), y(tr), 'NumNeighbors',1, ...
                              'Distance','euclidean','Standardize',false);
            case 'svm'
                mdl = fitcsvm(Xz(tr,:), y(tr), 'KernelFunction','rbf', ...
                              'KernelScale','auto','BoxConstraint',1, ...
                              'Standardize',false,'ClassNames',categories(y));
        end

        yhat = predict(mdl, Xz(te,:));
        if ~iscategorical(yhat), yhat = categorical(yhat, categories(y)); end

        cm = confusionmat(y(te), yhat, 'Order', categories(y));
        TP = diag(cm); FN = sum(cm,2)-TP; FP = sum(cm,1)'-TP; TN = sum(cm,'all')-TP-FN-FP;
        sens = TP ./ max(TP+FN,eps);
        spec = TN ./ max(TN+FP,eps);
        balAcc(k) = mean((sens+spec)/2);
        prec = TP ./ max(TP+FP,eps); rec = sens;
        f1 = 2*prec.*rec ./ max(prec+rec,eps);
        macroF1(k) = mean(f1,'omitnan');

        fprintf('Fold %d -> Bal.Acc: %.2f%% | Macro-F1: %.2f%%\n', k, 100*balAcc(k), 100*macroF1(k));
    end

    fprintf('\n>>> %s (5-Fold GROUP CV)\n', upper(modelName));
    fprintf('Mean Balanced Acc: %.2f%% ± %.2f\n', 100*mean(balAcc), 100*std(balAcc));
    fprintf('Mean Macro-F1    : %.2f%% ± %.2f\n', 100*mean(macroF1), 100*std(macroF1));
end
