%% ===================== ÖZELLİK ÇIKARIMI - TRAIN (BALANCED_V3_NOISE) =====================
clear; clc;

root        = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V3_NOISE";
trainPosDir = fullfile(root, "train", "pos");
trainNegDir = fullfile(root, "train", "neg");

fprintf("Train POS klasörü: %s\n", trainPosDir);
fprintf("Train NEG klasörü: %s\n", trainNegDir);

srTarget = 16000;
winDur   = 0.025;
hopDur   = 0.010;
NUM_MFCC = 13;
NUM_MEL  = 64;

Tpos_tr = extract_features_from_folder(trainPosDir, 'pos', srTarget, NUM_MFCC, NUM_MEL, winDur, hopDur);
Tneg_tr = extract_features_from_folder(trainNegDir, 'neg', srTarget, NUM_MFCC, NUM_MEL, winDur, hopDur);

T_train = [Tpos_tr; Tneg_tr];
fprintf("TRAIN toplam örnek: %d (pos=%d, neg=%d)\n", ...
    height(T_train), sum(T_train.LabelCat=="pos"), sum(T_train.LabelCat=="neg"));

% ---- CL için kolon seçimi (F_* + LabelCat) ----
fcols    = startsWith(T_train.Properties.VariableNames, "F_");
idxLabel = strcmp(T_train.Properties.VariableNames, "LabelCat");
colIdx   = fcols | idxLabel;

T_train_cl = T_train(:, colIdx);

save(fullfile(root, "T_train.mat"), "T_train");
save(fullfile(root, "T_train_cl.mat"), "T_train_cl");

fprintf("Kaydedildi: %s\n", fullfile(root, "T_train.mat"));
fprintf("Kaydedildi: %s\n", fullfile(root, "T_train_cl.mat"));

return;

%% ============================ LOCAL FUNCTION =============================

function T = extract_features_from_folder(folderPath, labelStr, srTarget, NUM_MFCC, NUM_MEL, winDur, hopDur)
    ads = audioDatastore(folderPath, 'IncludeSubfolders', false, 'FileExtensions', '.wav');
    n   = numel(ads.Files);
    fprintf("\n[%s] klasöründe %d dosya bulundu.\n", labelStr, n);

    feats  = cell(n,1);
    labels = cell(n,1);
    files  = strings(n,1);

    for i = 1:n
        src = ads.Files{i};
        try
            [x, fs] = audioread(src);
            if size(x,2)>1, x = mean(x,2); end
            if fs ~= srTarget
                x = resample(x, srTarget, fs);
                fs = srTarget;
            end
            x = unit_peak(x);

            feat = extract_157d(x, fs, NUM_MFCC, NUM_MEL, winDur, hopDur);

            feats{i}  = single(feat);
            labels{i} = labelStr;
            files(i)  = string(src);
        catch ME
            warning("Atlandı: %s -> %s", src, ME.message);
            feats{i}  = nan(1,157,'single');
            labels{i} = labelStr;
            files(i)  = string(src);
        end

        if mod(i, max(1,floor(n/10)))==0
            fprintf("[%s] %d/%d\n", labelStr, i, n);
        end
    end

    X = vertcat(feats{:});
    bad = any(~isfinite(X),2);
    if any(bad)
        warning("[%s] NaN/Inf içeren %d örnek atıldı.", labelStr, nnz(bad));
        X(bad,:)     = [];
        files(bad)   = [];
        labels(bad)  = [];
    end

    fn    = "F_" + (1:size(X,2));
    T     = array2table(X, 'VariableNames', fn);
    label = categorical(labels(:), {'neg','pos'}, 'Ordinal', false);
    if labelStr=="pos"
        label(:) = categorical("pos", {'neg','pos'});
    else
        label(:) = categorical("neg", {'neg','pos'});
    end
    T.LabelCat = label;
    T.File     = files;
end

function x = unit_peak(x)
    pk = max(abs(x));
    if pk>0, x = x/pk; end
end

function feat = extract_157d(x, fs, NUM_MFCC, NUM_MEL, winDur, hopDur)
    % MFCC
    C = mfcc(x, fs, 'NumCoeffs', NUM_MFCC);
    if size(C,1)<size(C,2), C = C.'; end
    mf_m = mean(C,1);
    mf_s = std(C,0,1);

    % log-Mel
    wl  = max(32, round(winDur*fs));
    hl  = max(16, round(hopDur*fs));
    win = hamming(wl, 'periodic');
    S   = melSpectrogram(x, fs, 'Window', win, ...
        'OverlapLength', wl-hl, 'NumBands', NUM_MEL, 'FFTLength', 1024);
    Slog = log10(S + 1e-12);
    ml_m = mean(Slog,2).';
    ml_s = std(Slog,0,2).';

    % ZCR / RMS / Spectral centroid
    zcr = mean(zerocrossrate(x));
    rv  = rms(x);
    sc  = mean(spectralCentroid(x,fs));

    feat = [mf_m, mf_s, ml_m, ml_s, zcr, rv, sc];
end
