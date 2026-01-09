%% ===================== VAL NOISE ROBUSTNESS EVALUATION =====================
% T_val üstünde SNR={20,10,5} dB gürültü overlay ile test.
% Gürültü kaynağı: NEG havuzundan 1 sn crop (domain-realistic).
% ÇIKTILAR:
%   - metrics_val_noise_SNRxx.txt (özet)
%   - val_noise_SNRxx_misclassified.csv (yanlışlar)
% NOT: Orijinal val dosyaları DEĞİŞTİRİLMEZ; özellik çıkarımı RAM'de yapılır.

clear; clc; rng(42);

%% === Kullanıcı yolları (gerekirse düzenle) ===
root     = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V2";
negDir   = fullfile(root, "val", "neg");     % Noise kaynağı için NEG havuzu (daha geniş istiyorsan 0 klasörünü ver)
srTarget = 16000;                             % Sabit örnekleme hızı
snrSet   = [20 10 5];                         % dB
winDur   = 0.025; hopDur = 0.010;             % Özellik pencereleri
NUM_MFCC = 13; NUM_MEL = 64;

%% === Verileri yükle
load(fullfile(root,"T_val.mat"), "T_val");  % orijinal val tablosu (File + LabelCat içerir)
assert(ismember("File", T_val.Properties.VariableNames), "T_val.File yok!");
assert(ismember("LabelCat", T_val.Properties.VariableNames), "T_val.LabelCat yok!");

% Noise kaynağı havuzu
adsNoise = audioDatastore(negDir, 'IncludeSubfolders', true, 'FileExtensions', '.wav');
assert(~isempty(adsNoise.Files), "Noise havuzu boş veya yol hatalı: %s", negDir);

%% === Modeli seç (export ettiğin CL modeli)
[fn, fp] = uigetfile('*.mat','Classification Learner’dan export edilen modeli seç');
assert(~isequal(fn,0), 'Model seçilmedi.');
S   = load(fullfile(fp,fn));
mfn = string(fieldnames(S));
mdl = S.(mfn(1));   % tek model varsayımı

%% === SNR döngüsü: her SNR için özellik çıkar + test et
for snr_db = snrSet
    fprintf('\n========== SNR = %d dB ==========\n', snr_db);
    % 1) Özellik çıkarımı (157D) - noise overlay ile
    [X, ytrue, usedFiles] = extract_val_features_with_noise(T_val, adsNoise, ...
        srTarget, snr_db, NUM_MFCC, NUM_MEL, winDur, hopDur);

    % 2) Tahmin (struct/predictFcn farklarına dayanıklı)
    [yhat, score] = safe_predict(mdl, X);

    % 3) Metrikler
    [cm, acc, prec, rec, f1] = metrics(ytrue, yhat);
    disp(array2table(cm, 'VariableNames',{'pred_neg','pred_pos'}, ...
                          'RowNames',   {'true_neg','true_pos'}));
    fprintf("SNR=%2d dB  Acc=%.3f  Prec=%.3f  Rec=%.3f  F1=%.3f  (n=%d)\n", ...
            snr_db, acc, prec, rec, f1, sum(cm,'all'));

    % 4) Yanlışlar → CSV
    wrong = (ytrue ~= yhat);
    outCsv = fullfile(root, sprintf("val_noise_SNR%d_misclassified.csv", snr_db));
    writetable( table(string(usedFiles(wrong)), string(ytrue(wrong)), string(yhat(wrong)), ...
                'VariableNames', {'File','True','Pred'}), outCsv);
    fprintf("Misclassified CSV: %s\n", outCsv);

    % 5) Özet → TXT append
    outTxt = fullfile(root, "metrics_val_noise.txt");
    fid = fopen(outTxt, 'a');
    if fid>0
        fprintf(fid, "SNR=%d dB\n", snr_db);
        fprintf(fid, "Confusion:\n[%4d %4d; %4d %4d]\n", cm(1,1), cm(1,2), cm(2,1), cm(2,2));
        fprintf(fid, "Acc=%.3f  Prec=%.3f  Rec=%.3f  F1=%.3f\n\n", acc, prec, rec, f1);
        fclose(fid);
    end
end
disp("Bitti. Özet: " + fullfile(root, "metrics_val_noise.txt"));

return;
%% ============================= LOCAL FUNCTIONS =============================

function [X, ytrue, files] = extract_val_features_with_noise(T_val, adsNoise, srTarget, snr_db, NUM_MFCC, NUM_MEL, winDur, hopDur)
    n = height(T_val);
    feats = cell(n,1); ytrue = T_val.LabelCat; files = string(T_val.File);
    for i = 1:n
        try
            [x,fs] = audioread(T_val.File{i});
            if size(x,2)>1, x = mean(x,2); end
            if fs ~= srTarget, x = resample(x, srTarget, fs); fs = srTarget; end
            x = unit_pk(x);

            % 1 sn crop (deterministik değil; robustluk için random crop makul)
            x1 = one_sec_crop(x, fs);

            % noise: neg havuzundan 1 dosya seç → 1 sn crop
            nfile = adsNoise.Files{ randi(numel(adsNoise.Files)) };
            [n,fsn] = audioread(nfile);
            if size(n,2)>1, n = mean(n,2); end
            if fsn ~= fs, n = resample(n, fs, fsn); end
            n1 = one_sec_crop(n, fs);

            % SNR hedefli miksaj
            x_noisy = mix_snr(x1, n1, snr_db);

            % 157D özellik
            feats{i} = single(extract_157d(x_noisy, fs, NUM_MFCC, NUM_MEL, winDur, hopDur));
        catch ME
            warning("Atlandı: %s -> %s", T_val.File{i}, ME.message);
            feats{i} = nan(1,157,'single');
        end
        if mod(i, max(1,floor(n/10)))==0
            fprintf("SNR=%2d: %3d/%3d\n", snr_db, i, n);
        end
    end
    X = vertcat(feats{:});
    bad = any(~isfinite(X),2);
    X(bad,:) = []; ytrue(bad) = []; files(bad) = [];
end

function x = unit_pk(x)
    pk = max(abs(x)); if pk>0, x = x/pk; end
end

function x1 = one_sec_crop(x, fs)
    L = fs;
    if numel(x) < L
        x1 = [x; zeros(L-numel(x),1)];
    else
        s0 = randi([1, numel(x)-L+1]);
        x1 = x(s0:s0+L-1);
    end
end

function y = mix_snr(x, n, snr_db)
    % x = hedef sinyal (1 sn), n = noise (1 sn)
    Ps = mean(x.^2) + 1e-12;
    Pn_tgt = Ps / (10^(snr_db/10));
    Pn = mean(n.^2) + 1e-12;
    n_scaled = n * sqrt(Pn_tgt / Pn);
    y = x + n_scaled;
    % Soft clipping
    y = max(min(y, 1), -1);
end

function feat = extract_157d(x, fs, NUM_MFCC, NUM_MEL, winDur, hopDur)
    % MFCC (13) → mean+std (26)
    C = mfcc(x, fs, 'NumCoeffs', NUM_MFCC);
    if size(C,1)<size(C,2), C = C.'; end
    mf_m = mean(C,1); mf_s = std(C,0,1);

    % log-Mel (64) → mean+std (128)
    wl = max(32, round(winDur*fs));
    hl = max(16, round(hopDur*fs));
    win = hamming(wl,'periodic');
    S = melSpectrogram(x, fs, 'Window', win, 'OverlapLength', wl-hl, ...
                       'NumBands', NUM_MEL, 'FFTLength', 1024);
    Slog = log10(S + 1e-12);
    ml_m = mean(Slog,2).'; ml_s = std(Slog,0,2).';

    % ZCR / RMS / Spectral Centroid (3)
    zcr = mean(zerocrossrate(x));
    rv  = rms(x);
    sc  = mean(spectralCentroid(x,fs));

    feat = [mf_m, mf_s, ml_m, ml_s, zcr, rv, sc];  % 26 + 128 + 3 = 157
end

function [yhat, score] = safe_predict(mdl, X)
    score = [];
    try
        [yhat, score] = predict(mdl, X);
    catch
        try
            % mdl.predictFcn tablo bekleyebilir
            T = array2table(X, 'VariableNames', "F_"+(1:size(X,2)));
            [yhat, score] = mdl.predictFcn(T);
        catch
            T = array2table(X, 'VariableNames', "F_"+(1:size(X,2)));
            yhat = mdl.predictFcn(T);
        end
    end
    yhat = categorical(yhat);
end

function [cm, acc, prec, rec, f1] = metrics(ytrue, yhat)
    order = categorical({'neg','pos'});
    cm = confusionmat(ytrue, yhat, 'Order', order);
    TN=cm(1,1); FP=cm(1,2); FN=cm(2,1); TP=cm(2,2);
    prec = TP / max(1, TP+FP);
    rec  = TP / max(1, TP+FN);
    f1   = 2*prec*rec / max(1e-12, prec+rec);
    acc  = (TP+TN) / sum(cm,'all');
end
