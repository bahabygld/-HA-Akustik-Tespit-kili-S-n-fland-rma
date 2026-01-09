%% === VAL ÜZERİNDE TEST (modeli diyalogla seç) ===
clear; clc;

root = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V2";
load(fullfile(root,"T_val_cl.mat"),"T_val_cl");

% Model .mat dosyasını seç
[fn, fp] = uigetfile('*.mat','Classification Learner’dan export ettiğin modeli seç');
assert(~isequal(fn,0), 'Model seçilmedi.');
modelPath = fullfile(fp, fn);

% Modeli yükle (değişken adını otomatik tespit et)
S  = load(modelPath);
fnv = string(fieldnames(S));
mdl = S.(fnv(1));

% Özellik matrisi ve etiket
fcols = startsWith(T_val_cl.Properties.VariableNames,"F_");
X     = T_val_cl{:, fcols};
ytrue = T_val_cl.LabelCat;

% Tahmin + skor
try
    [yhat, score] = predict(mdl, X);
catch
    try
        [yhat, score] = mdl.predictFcn(T_val_cl(:, fcols));
    catch
        yhat = mdl.predictFcn(T_val_cl(:, fcols));
        score = [];
    end
end
yhat = categorical(yhat);

% Metrikler
order = categorical({'neg','pos'});
cm = confusionmat(ytrue, yhat, 'Order', order);
TN=cm(1,1); FP=cm(1,2); FN=cm(2,1); TP=cm(2,2);
prec = TP/max(1,TP+FP); rec = TP/max(1,TP+FN);
f1 = 2*prec*rec/max(1e-12,prec+rec); acc=(TP+TN)/sum(cm,'all');
disp(array2table(cm,'VariableNames',{'pred_neg','pred_pos'},'RowNames',{'true_neg','true_pos'}));
fprintf("Acc=%.3f  Prec=%.3f  Rec=%.3f  F1=%.3f\n", acc,prec,rec,f1);
