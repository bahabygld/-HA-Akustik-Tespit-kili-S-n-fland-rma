% ===============================================
% prefix_length_test.m
% Amaç: Farklı prefix uzunlukları (4–7) için
%       group-based 5-fold CV performansını karşılaştırmak
% ===============================================
clc; clear; close all;

% --- Veri Yükleme ---
matFile = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\İha Akustik ses\Veri_Setleri\ozellik_cikarimi\dads_features_light.mat";
S = load(matFile);
assert(isfield(S,'T_clip_light'), 'T_clip_light bulunamadı.');
T = S.T_clip_light;

if ~iscategorical(T.label)
    T.label = categorical(T.label);
end

[~,fn,~] = cellfun(@fileparts, T.file, 'UniformOutput', false);
fn = string(fn);
parts = split(T.file, filesep);
folderNames = parts(:, end-1);  % '0' veya '1'

feats = startsWith(T.Properties.VariableNames, 'F_');
X = T{:,feats};
y = T.label;

% --- Denenecek prefix uzunlukları ---
prefixLens = 4:7;
K = 5;

results = table('Size',[numel(prefixLens) 4], ...
    'VariableTypes', {'double','double','double','double'}, ...
    'VariableNames', {'PrefixLen','MeanBalAcc','MeanMacroF1','NumGroups'});

for pi = 1:numel(prefixLens)
    prefixN = prefixLens(pi);
    prefix = extractBefore(fn, prefixN+1); % örn. 0016 (prefixN=4)
    G = folderNames + "_" + prefix;

    % Grup sayısı ve sınıf dengesi kontrol
    numGroups = numel(unique(G));
    results.NumGroups(pi) = numGroups;
    results.PrefixLen(pi) = prefixN;

    try
        cv = cvpartition(G,'KFold',K);
    catch
        warning("cvpartition hatası: prefix=%d, grup sayısı=%d", prefixN, numGroups);
        continue
    end

    balAcc = zeros(K,1);
    macroF1 = zeros(K,1);

    for k = 1:K
        trainIdx = training(cv,k);
        testIdx  = test(cv,k);

        % --- Standartizasyon ---
        mu = mean(X(trainIdx,:));
        sig = std(X(trainIdx,:));
        Xz = (X - mu) ./ max(sig,1e-8);

        % --- Basit SVM (Gaussian) ---
        mdl = fitcsvm(Xz(trainIdx,:), y(trainIdx), ...
                      'KernelFunction','rbf', 'KernelScale',1, ...
                      'BoxConstraint',1, 'Standardize',false);

        yhat = predict(mdl, Xz(testIdx,:));
        if ~iscategorical(yhat)
            yhat = categorical(yhat, categories(y));
        end

        cm = confusionmat(y(testIdx), yhat, 'Order', categories(y));
        TP = diag(cm); FN = sum(cm,2)-TP; FP = sum(cm,1)'-TP; TN = sum(cm,'all')-TP-FN-FP;
        sens = TP ./ max(TP+FN,eps);
        spec = TN ./ max(TN+FP,eps);
        balAcc(k) = mean((sens+spec)/2);
        prec = TP ./ max(TP+FP,eps);
        rec = sens;
        f1 = 2*prec.*rec ./ max(prec+rec,eps);
        macroF1(k) = mean(f1,'omitnan');
    end

    results.MeanBalAcc(pi) = mean(balAcc);
    results.MeanMacroF1(pi)= mean(macroF1);
end

% --- Sonuçları yazdır ---
disp('===============================================')
disp('Prefix Uzunluğu Test Sonuçları (5-Fold Group CV)')
disp('===============================================')
disp(results)

% --- Grafik ---
figure;
yyaxis left
plot(results.PrefixLen, results.MeanBalAcc*100, '-o','LineWidth',1.5);
ylabel('Balanced Accuracy (%)');
yyaxis right
plot(results.PrefixLen, results.NumGroups,'--s','LineWidth',1);
ylabel('Grup Sayısı');
xlabel('Prefix Uzunluğu (dosya adı karakter sayısı)');
grid on;
title('Prefix uzunluğunun performansa etkisi');
