%% ===================== DRONE VAR/YOK - DENGELİ + NOISE TRAIN (v3) =====================
% TRAIN-Pos örneklerinin bir kısmına (p_noise) SNR{20,15,10,7,5} dB noise overlay.
% VAL temiz kalır.
%
% ÇIKTI DİZİN YAPISI:
%   C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V3_NOISE\
%       train\pos
%       train\neg
%       val\pos
%       val\neg

clear; clc; rng(42);

%% ===================== PARAMETRELER =====================
basePath ="C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V2\train";

posDir  = fullfile(basePath, "pos");  % pozitif kaynak
negDir  = fullfile(basePath, "neg");                  % negatif kaynak

outRoot = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V3_NOISE";

valRatioPos    = 0.30;   % pozitiflerin %30'u val
targetPosTrain = 3000;   % toplam TRAIN poz sayısı

srTarget = 16000;

% Noise parametreleri (sadece TRAIN-POS)
p_noise  = 0.60;                         % TRAIN-POS örneklerinin %60'ına noise
snr_vals = [20 15 10 7 5];               % dB
snr_prob = [0.25 0.25 0.25 0.15 0.10];   % olasılıklar
snr_prob = snr_prob / sum(snr_prob);     % normalize

%% ===================== DATASTORE'LAR =====================
adsPos = audioDatastore(posDir, 'IncludeSubfolders', true, 'FileExtensions', '.wav');
adsNeg = audioDatastore(negDir, 'IncludeSubfolders', true, 'FileExtensions', '.wav');
nPos   = numel(adsPos.Files);
nNeg   = numel(adsNeg.Files);

fprintf("Pozitif=%d, Negatif=%d\n", nPos, nNeg);
if nPos < 50
    error("Pozitif dosya sayısı çok az görünüyor. posDir'i kontrol et.");
end

% Noise kaynağı: yine büyük 0 havuzu
adsNegNoise = audioDatastore(negDir, 'IncludeSubfolders', true, 'FileExtensions', '.wav');
if isempty(adsNegNoise.Files)
    error("Noise kaynağı için NEG havuzu boş: %s", negDir);
end

% Çıkış klasörleri
outPosTr = fullfile(outRoot,"train","pos");
outPosVa = fullfile(outRoot,"val","pos");
outNegTr = fullfile(outRoot,"train","neg");
outNegVa = fullfile(outRoot,"val","neg");

mkdir(outPosTr); mkdir(outPosVa);
mkdir(outNegTr); mkdir(outNegVa);

fprintf("=== ÇIKTI KLASÖRLERİ ===\n%s\n%s\n%s\n%s\n", ...
    outPosTr, outPosVa, outNegTr, outNegVa);

%% ===================== 1) POZİTİF: VAL / TRAIN AYIR =====================
idx = randperm(nPos);
nVal = max(1, round(nPos * valRatioPos));

posValFiles  = string(adsPos.Files(idx(1:nVal)));
posTrainOrig = string(adsPos.Files(idx(nVal+1:end)));

% VAL POS (her biri 1 sn, noise yok)
for i = 1:numel(posValFiles)
    [x, fs] = read_wav_mono(posValFiles(i), srTarget);
    seg     = one_sec_crop(x, fs);
    fn      = fullfile(outPosVa, sprintf("pos_val_%05d.wav", i));
    audiowrite(fn, seg, fs);
end
fprintf("VAL POS yazıldı: %d\n", numel(posValFiles));

%% ===================== 2) TRAIN POS: CLEAN + AUGMENT + NOISE =====================
posTrainList = strings(0,1);

% 2a) Kaynaklardan direkt 1 sn crop (clean + maybe noise)
for i = 1:numel(posTrainOrig)
    [x, fs] = read_wav_mono(posTrainOrig(i), srTarget);
    seg     = one_sec_crop(x, fs);
    seg     = unit_peak(seg);
    seg     = maybe_add_noise(seg, fs, adsNegNoise, p_noise, snr_vals, snr_prob);
    fn      = fullfile(outPosTr, sprintf("pos_src_%05d.wav", i));
    audiowrite(fn, seg, fs);
    posTrainList(end+1) = fn; %#ok<SAGROW>
end

% 2b) Augment + noise ile hedef sayıya tamamla
augNeeded = max(0, targetPosTrain - numel(posTrainList));
fprintf("Augment üretilecek adet: %d (hedef TRAIN POS=%d)\n", augNeeded, targetPosTrain);

for k = 1:augNeeded
    src = posTrainOrig(randi(numel(posTrainOrig)));
    [x, fs] = read_wav_mono(src, srTarget);
    a       = one_sec_crop(x, fs);

    % Time-stretch ±%8
    stretch_factor = 0.92 + 0.16*rand(); % 0.92–1.08
    a = aug_timescale(a, stretch_factor);

    % %80 ihtimalle ±1..2 semiton pitch shift
    if rand < 0.8
        sem = randsample([-2 -1 1 2], 1);
        a   = aug_pitch_shift_semitone(a, fs, sem);
    end

    a  = unit_peak(a);
    a  = maybe_add_noise(a, fs, adsNegNoise, p_noise, snr_vals, snr_prob);
    fn = fullfile(outPosTr, sprintf("pos_aug_%05d.wav", k));
    audiowrite(fn, a, fs);
    posTrainList(end+1) = fn; %#ok<SAGROW>
end

fprintf("Toplam TRAIN POS: %d\n", numel(posTrainList));

%% ===================== 3) NEG: VAL / TRAIN DENGELE =====================
% VAL NEG sayısı = VAL POS sayısı
nNegVal = numel(posValFiles);
negIdx  = randperm(nNeg);

negValFiles  = string(adsNeg.Files(negIdx(1:nNegVal)));
negTrainPool = string(adsNeg.Files(negIdx(nNegVal+1:end)));

% TRAIN NEG sayısı = TRAIN POS sayısı
needNegTrain = numel(posTrainList);
if numel(negTrainPool) < needNegTrain
    warning("Negatif train havuzu yetersiz (%d). Tümü kullanılacak.", numel(negTrainPool));
    needNegTrain = numel(negTrainPool);
end
negTrainSel = negTrainPool(randperm(numel(negTrainPool), needNegTrain));

% VAL NEG (clean)
for i = 1:numel(negValFiles)
    [x, fs] = read_wav_mono(negValFiles(i), srTarget);
    seg     = one_sec_crop(x, fs);
    fn      = fullfile(outNegVa, sprintf("neg_val_%05d.wav", i));
    audiowrite(fn, seg, fs);
end

% TRAIN NEG (clean)
for i = 1:numel(negTrainSel)
    [x, fs] = read_wav_mono(negTrainSel(i), srTarget);
    seg     = one_sec_crop(x, fs);
    fn      = fullfile(outNegTr, sprintf("neg_train_%06d.wav", i));
    audiowrite(fn, seg, fs);
end

fprintf("ÖZET → VAL POS: %d | VAL NEG: %d | TRAIN POS: %d | TRAIN NEG: %d\n", ...
    numel(posValFiles), numel(negValFiles), numel(posTrainList), numel(negTrainSel));

return;

%% ============================ LOCAL FONKSİYONLAR =============================

function [x, fs] = read_wav_mono(path, srTarget)
    [y, fs] = audioread(path);
    if size(y,2)>1, y = mean(y,2); end
    if fs ~= srTarget
        y = resample(y, srTarget, fs); fs = srTarget;
    end
    x = y;
end

function seg = one_sec_crop(x, fs)
    L = fs;
    if numel(x) < L
        seg = [x; zeros(L-numel(x),1)];
    else
        s0  = randi([1, numel(x)-L+1]);
        seg = x(s0:s0+L-1);
    end
end

function x = unit_peak(x)
    pk = max(abs(x));
    if pk > 0, x = x/pk; end
end

function x2 = aug_timescale(x, factor)
    L  = numel(x);
    x2 = resample(x, round(1000*factor), 1000);
    if numel(x2) >= L
        x2 = x2(1:L);
    else
        x2(end+1:L) = 0;
    end
end

function x2 = aug_pitch_shift_semitone(x, fs, semitones)
    factor = 2^(semitones/12);
    y      = resample(x, round(1000*factor), 1000);
    x2     = resample(y, fs, fs);
    L      = fs;
    if numel(x2) >= L
        x2 = x2(1:L);
    else
        x2(end+1:L) = 0;
    end
end

function seg_out = maybe_add_noise(seg_in, fs, adsNegNoise, p_noise, snr_vals, snr_prob)
    if rand >= p_noise
        seg_out = seg_in;
        return;
    end
    idx   = randi(numel(adsNegNoise.Files));
    nFile = adsNegNoise.Files{idx};
    [n, fsn] = audioread(nFile);
    if size(n,2)>1, n = mean(n,2); end
    if fsn ~= fs
        n = resample(n, fs, fsn);
    end
    n1     = one_sec_crop(n, fs);
    snr_db = randsample(snr_vals, 1, true, snr_prob);
    seg_out = mix_snr(seg_in, n1, snr_db);
end

function y = mix_snr(x, n, snr_db)
    Ps = mean(x.^2) + 1e-12;
    Pn = mean(n.^2) + 1e-12;
    Pn_tgt = Ps / (10^(snr_db/10));
    alpha  = sqrt(Pn_tgt / Pn);
    y      = x + alpha*n;
    y      = max(min(y,1),-1);
end
