%% test_balanced_v3_son_test.m
clc;
clear;

%% 1) TEST SETI KLASORU
rootTest = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V3_SON_TEST";
fprintf("Test klasoru : %s\n", rootTest);

%% 2) MODEL DOSYASINI SEC (CL'DEN EXPORT EDILMIS .MAT)
[fn, fp] = uigetfile('*.mat', 'CL''den export edilen modeli sec');
assert(~isequal(fn,0), "Model secilmedi, iptal edildi.");

modelPath = fullfile(fp, fn);
fprintf("Secilen model dosyasi: %s\n", modelPath);

S   = load(modelPath);
fns = fieldnames(S);

mdl = [];
mdlName = '';
for i = 1:numel(fns)
    val = S.(fns{i});
    if isstruct(val) && isfield(val, "predictFcn") && isfield(val, "RequiredVariables")
        mdl     = val;
        mdlName = fns{i};
        break;
    end
end

if isempty(mdl)
    error("Secilen .mat dosyasinda predictFcn + RequiredVariables iceren bir CL modeli yok.");
end

fprintf("Model degiskeni olarak '%s' kullaniliyor.\n", mdlName);

varNames = mdl.RequiredVariables;
fprintf("Modelin bekledigi ozellik sayisi: %d\n", numel(varNames));

%% 3) 0 VE 1 KLASORLERINDEN AUDIO DATASTORE OLUSTUR

adsTest = audioDatastore(rootTest, ...
    "IncludeSubfolders", true, ...
    "FileExtensions", ".wav", ...
    "LabelSource", "foldernames");

nTest = numel(adsTest.Files);
fprintf("\nTest setindeki toplam dosya sayisi: %d\n", nTest);

% Klasor isimlerinden gelen etiketler: '0' ve '1'
yFolderLabels = adsTest.Labels;   % categorical ('0','1')
fprintf("Klasor etiket dagilimi:\n");
disp(tabulate(yFolderLabels));

%% 4) ORNEK BIR DOSYADAN OZELLIK CIKAR VE BOYUTU OGREN

fprintf("\nOrnek bir dosyadan ozellik cikartiliyor...\n");
exampleFile = adsTest.Files{1};
featExample = extract_drone_features(exampleFile);   % daha once yazdigimiz fonksiyon
featExample = featExample(:).';

D = numel(featExample);
fprintf("Ozellik vektor boyutu (D) : %d\n", D);

%% 5) TUM DOSYALAR ICIN OZELLIK MATRISI OLUSTUR

X = zeros(nTest, D, "single");
X(1, :) = featExample;

for i = 2:nTest
    fpath = adsTest.Files{i};
    feat  = extract_drone_features(fpath);
    feat  = feat(:).';
    
    if numel(feat) ~= D
        error("Dosya %d icin ozellik boyutu farkli! (%d yerine %d)", ...
            i, D, numel(feat));
    end
    
    X(i, :) = feat;
    
    if mod(i, max(1,floor(nTest/10))) == 0
        fprintf("%d / %d dosya islenmis durumda...\n", i, nTest);
    end
end

fprintf("Tum dosyalar icin ozellik cikarma tamam.\n");

%% 6) TABLO OLUSTUR VE MODELE VER

if numel(varNames) ~= D
    error("Model %d ozellik bekliyor, bizim ozellik boyutu %d. Egitim pipeline'i ile test pipeline'i farkli.", ...
        numel(varNames), D);
end

T_test = array2table(X, "VariableNames", varNames);

%% 7) TAHMIN AL

yPred = mdl.predictFcn(T_test);   % genelde categorical('neg','pos') dondurur

% Model etiketleri (neg/pos) ile klasor etiketlerini eslemek icin,
% klasor etiketlerinden 'neg'/'pos' kategorisi uretelim:
%  - 0 klasoru -> 'neg'
%  - 1 klasoru -> 'pos'
yTrueModel = categorical(repmat("neg", nTest, 1), ["neg","pos"]);
yTrueModel(yFolderLabels == '1') = categorical("pos", ["neg","pos"]);

if ~iscategorical(yPred)
    yPred = categorical(yPred);
end

%% 8) KARISIKLIK MATRISI VE METRIKLER

order = categorical({'neg','pos'});
cm = confusionmat(yTrueModel, yPred, 'Order', order);

disp("Confusion matrix [true x predicted] (siralar: neg,pos):");
cmTable = array2table(cm, ...
    'VariableNames', {'pred_neg','pred_pos'}, ...
    'RowNames', {'true_neg','true_pos'});
disp(cmTable);

tn = cm(1,1);
fp = cm(1,2);
fn = cm(2,1);
tp = cm(2,2);

N    = sum(cm,'all');
acc  = (tp + tn) / N;
sens = tp / (tp + fn + eps);   % pos (drone var) recall
spec = tn / (tn + fp + eps);   % neg (drone yok) recall

fprintf("N = %d ornek\n", N);
fprintf("Dogrluk (Accuracy)              : %.2f %%\n",  acc*100);
fprintf("Duyarlilik (Sensitivity, pos=1) : %.2f %%\n",  sens*100);
fprintf("Ozgunluk  (Specificity, neg=0)  : %.2f %%\n",  spec*100);
