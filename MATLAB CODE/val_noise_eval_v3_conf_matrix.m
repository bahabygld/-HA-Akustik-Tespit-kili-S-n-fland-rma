clear; clc; close all;
root = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\1.Sınıf 2.Dönem\İha Akustik ses\SON DENGELENMİŞ\BALANCED_V3_SON_TEST_FIXED";
valPosDir = fullfile(root,"1");
valNegDir = fullfile(root,"0");

srTarget = 16000;
NUM_MFCC = 13;
NUM_MEL  = 64;
winDur   = 0.025;
hopDur   = 0.010;

snrList = [20 15 10 5];  % Bildiri için 4 senaryo
modelPath = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\1.Sınıf 2.Dönem\İha Akustik ses\MATLAB DATA\trainedModelV3_SON.mat";  % <-- DÜZELT
outDir = fullfile(root, "OUT_SNR_CM");
if ~exist(outDir, "dir"), mkdir(outDir); end

S = load(modelPath);
mfn = string(fieldnames(S));
mdl = S.(mfn(1));
fprintf('Seçilen model: %s\n', mfn(1));

% rng(0);  % her çalıştırmada aynı kırpma olsun

order = categorical({'neg','pos'});

for snrDB = snrList
    fprintf('\n==============================\n');
    fprintf('   SNR = %d dB için test\n', snrDB);
    fprintf('==============================\n');

    T_val_noise = build_noisy_val_table( ...
        valPosDir, valNegDir, snrDB, ...
        srTarget, NUM_MFCC, NUM_MEL, winDur, hopDur);

    fcols = startsWith(T_val_noise.Properties.VariableNames,"F_");
    Xval  = T_val_noise(:, fcols);
    ytrue = T_val_noise.LabelCat;

    yhat = mdl.predictFcn(Xval);

    cm = confusionmat(ytrue, yhat, 'Order', order);

    tn = cm(1,1); fp = cm(1,2);
    fn = cm(2,1); tp = cm(2,2);

    N = sum(cm,'all');
    acc  = (tp+tn) / N;
    prec = tp / (tp+fp + eps);
    rec  = tp / (tp+fn + eps);
    F1   = 2*prec*rec / (prec+rec + eps);

    disp(array2table(cm,'VariableNames',{'pred_neg','pred_pos'}, ...
                        'RowNames',{'true_neg','true_pos'}));
    fprintf('Acc=%.3f  Prec=%.3f  Rec=%.3f  F1=%.3f  (n=%d)\n', acc, prec, rec, F1, N);

    fig = figure('Visible','on'); %#ok<NASGU>
    cc = confusionchart(cm, {'neg','pos'});
    cc.Title = sprintf('Confusion Matrix (SNR = %d dB)', snrDB);
    cc.RowSummary = 'row-normalized';
    cc.ColumnSummary = 'column-normalized';

    pngName = fullfile(outDir, sprintf("Fig_SNR_%ddB_CM.png", snrDB));
    exportgraphics(gcf, pngName, 'Resolution', 300);

    figName = fullfile(outDir, sprintf("Fig_SNR_%ddB_CM.fig", snrDB));
    savefig(gcf, figName);

    
end

fprintf('\nBitti. Görseller: %s\n', outDir);

return;

function T = build_noisy_val_table(valPosDir, valNegDir, snrDB, ...
                                   srTarget, NUM_MFCC, NUM_MEL, winDur, hopDur)
    posFiles = dir(fullfile(valPosDir, "*.wav"));
    negFiles = dir(fullfile(valNegDir, "*.wav"));

    nPos = numel(posFiles);
    nNeg = numel(negFiles);

    if nPos==0 || nNeg==0
        error("VAL pos/neg klasörlerinde .wav bulunamadı.");
    end

    featList = {};
    labList  = {};

    for i=1:nPos
        fpath = fullfile(posFiles(i).folder, posFiles(i).name);
        [x,fs] = read_wav_mono(fpath, srTarget);
        x1     = one_sec_crop(x, fs);
        xn     = add_noise_snr(x1, snrDB);
        feat   = extract_feats(xn, fs, NUM_MFCC, NUM_MEL, winDur, hopDur);
        featList{end+1,1} = single(feat); %#ok<SAGROW>
        labList{end+1,1}  = categorical("pos");
    end

    for i=1:nNeg
        fpath = fullfile(negFiles(i).folder, negFiles(i).name);
        [x,fs] = read_wav_mono(fpath, srTarget);
        x1     = one_sec_crop(x, fs);
        xn     = add_noise_snr(x1, snrDB);
        feat   = extract_feats(xn, fs, NUM_MFCC, NUM_MEL, winDur, hopDur);
        featList{end+1,1} = single(feat); %#ok<SAGROW>
        labList{end+1,1}  = categorical("neg");
    end

    X = vertcat(featList{:});
    lab = vertcat(labList{:});

    fnames = "F_" + (1:size(X,2));
    T      = array2table(X, 'VariableNames', fnames);
    T.LabelCat = lab;
end

function [x, fs] = read_wav_mono(path, srTarget)
    [y,fs] = audioread(path);
    if size(y,2)>1, y = mean(y,2); end
    if fs ~= srTarget
        y = resample(y, srTarget, fs);
        fs = srTarget;
    end
    mx = max(abs(y));
    if mx>0, x = y/mx; else, x = y; end
end

function seg = one_sec_crop(x, fs)
    L = fs;
    if numel(x) < L
        seg = [x; zeros(L-numel(x),1)];
    else
        s0 = randi([1, numel(x)-L+1]);
        seg = x(s0:s0+L-1);
    end
end

function y = add_noise_snr(x, snrDB)
    x = x(:);
    Px = mean(x.^2);
    if Px == 0, y = x; return; end
    n  = randn(size(x));
    Pn = mean(n.^2);
    alpha = sqrt(Px / (Pn * 10^(snrDB/10)));
    y = x + alpha*n;
end

function feat = extract_feats(x, fs, NUM_MFCC, NUM_MEL, winDur, hopDur)
    C = mfcc(x, fs, 'NumCoeffs', NUM_MFCC);
    if size(C,1) < size(C,2), C = C.'; end
    mf_m = mean(C,1);
    mf_s = std (C,0,1);

    wl  = max(32, round(winDur*fs));
    hl  = max(16, round(hopDur*fs));
    win = hamming(wl,'periodic');
    S   = melSpectrogram(x, fs, ...
            'Window',win, ...
            'OverlapLength',wl-hl, ...
            'NumBands',NUM_MEL, ...
            'FFTLength',1024);
    Slog = log10(S + 1e-12);
    ml_m = mean(Slog,2).';
    ml_s = std (Slog,0,2).';

    zcr = mean(zerocrossrate(x));
    rv  = rms(x);
    sc  = mean(spectralCentroid(x,fs));

    feat = [mf_m, mf_s, ml_m, ml_s, zcr, rv, sc];
end

