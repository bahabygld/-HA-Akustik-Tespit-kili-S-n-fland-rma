%% test_balanced_v3_son_test_FINAL.m
clc; clear;

%% 1) PATHS
rootTest = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\1.Sınıf 2.Dönem\İha Akustik ses\SON DENGELENMİŞ\BALANCED_V3_SON_TEST_FIXED";
modelFile = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\1.Sınıf 2.Dönem\İha Akustik ses\MATLAB DATA\trainedModelV3_SON.mat";

fprintf("Test klasoru  : %s\n", rootTest);
fprintf("Model dosyasi : %s\n\n", modelFile);

%% 2) LOAD MODEL
S = load(modelFile);
fn = fieldnames(S);
mdl = S.(fn{1});

if ~isfield(mdl,"predictFcn")
    error("Model struct degil veya predictFcn yok!");
end

varNames = mdl.RequiredVariables;
D_model  = numel(varNames);
fprintf("Model %d ozellik bekliyor.\n\n", D_model);

%% 3) DATASTORE
ads = audioDatastore(rootTest, ...
    "IncludeSubfolders",true, ...
    "FileExtensions",".wav", ...
    "LabelSource","foldernames");

n = numel(ads.Files);
fprintf("Toplam test dosyasi: %d\n", n);

yTrueFolder = ads.Labels;

%% 4) Extract example feature
ex = extract_drone_features(ads.Files{1});
ex = ex(:).';
D_feat = numel(ex);

fprintf("Gercek ozellik boyutu: %d\n\n", D_feat);

if D_feat ~= D_model
    error("Model %d ozellik bekliyor ama extract_drone_features %d üretiyor!", ...
        D_model, D_feat);
end

%% 5) Tum dosyalar icin ozellik cikar
X = zeros(n, D_model, "single");
valid = true(n,1);

X(1,:) = ex;

for i = 2:n
    f = ads.Files{i};

    try
        feat = extract_drone_features(f);
        feat = feat(:).';
    catch
        valid(i) = false;
        continue;
    end

    if numel(feat) ~= D_model
        valid(i) = false;
        continue;
    end

    X(i,:) = feat;

    if mod(i, floor(n/10)) == 0
        fprintf("%d / %d islem tamam...\n", i, n);
    end
end

%% Temizle
X = X(valid,:);
yTrueFolder = yTrueFolder(valid);

fprintf("\nGecerli ornek sayisi: %d\n\n", size(X,1));

%% 6) Prediction table
T_test = array2table(X, "VariableNames", varNames);

%% Predicted labels
yPred = mdl.predictFcn(T_test);
if ~iscategorical(yPred)
    yPred = categorical(yPred);
end

%% True labels
yTrue = categorical(repmat("neg",size(X,1),1), ["neg","pos"]);
yTrue(string(yTrueFolder)=="1") = categorical("pos",["neg","pos"]);

%% 7) CONFUSION MATRIX
order = categorical(["neg","pos"],["neg","pos"]);
cm = confusionmat(yTrue, yPred, "Order", order);

fprintf("Confusion matrix:\n");
disp(array2table(cm, ...
    "VariableNames",{'pred_neg','pred_pos'}, ...
    "RowNames",{'true_neg','true_pos'}));

tn = cm(1,1); fp = cm(1,2);
fn = cm(2,1); tp = cm(2,2);

acc  = (tp+tn)/sum(cm,'all');
sens = tp/(tp+fn+eps);
spec = tn/(tn+fp+eps);

fprintf("Accuracy  : %.2f %%\n", acc*100);
fprintf("Sensitivity (pos): %.2f %%\n", sens*100);
fprintf("Specificity (neg): %.2f %%\n\n", spec*100);

%% ---- Confusion Chart ----
figure;
confusionchart(yTrue, yPred, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
title("Confusion Matrix (Normalized)");
ylabel("True Class");
xlabel("Predicted Class");

%% 8) ROC CURVE - sade ve doğru hali

if isempty(scores)
    warning("Skor bulunamadi! ROC hesabi atlandi.");
    return;
end

% Pozitif sınıf 'pos' için kolon seç
cls = string(subModel.ClassNames);
posCol = find(cls=="pos",1);
if isempty(posCol)
    posCol = 2;
    warning("'pos' sinif kolonu bulunamadi, 2. kolon pozitif sayildi.");
end

yTrueNum = double(yTrue=="pos");
scorePos = scores(:, posCol);

[fpRate, tpRate, Tthr, AUC] = perfcurve(yTrueNum, scorePos, 1);

fprintf("ROC icin kullanilan skor kolonu: %d\n", posCol);
fprintf("ROC AUC = %.4f\n", AUC);

figure;
plot(fpRate, tpRate, "LineWidth", 2);
grid on;
xlabel("False Positive Rate");
ylabel("True Positive Rate");
title(sprintf("ROC Curve (AUC = %.4f)", AUC));
