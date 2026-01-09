%% ===================== DRONE VAR/YOK - DENGELEME (SADECE WAV) =====================
% 1) Pozitiflerin %30'u VAL/TEST (augment YOK)
% 2) Kalan %70 TRAIN: ±%8 time-stretch + ±1..2 semiton pitch augment
% 3) Negatifler: VAL = POS_VAL sayısı, TRAIN = POS_TRAIN sayısı (her biri 1 sn)
% 4) Özellik çıkarımı YOK. Sadece WAV dosyaları üretilir.

clear; clc; rng(42);

%% ===================== KULLANICI PARAMETRELERİ =====================
% Kaynak klasörler (okuma tarafında Unicode sorun çıkarmaz)
basePath  = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\İha Akustik ses\Veri_Setleri\yeni";
posDir    = fullfile(basePath, "drone_var_yeni_v2");   % pozitif (drone var) kaynak WAV'lar
negDir    = fullfile(basePath, "0");                   % negatif (drone yok) kaynak WAV'lar

% *** ÇIKTI KÖKÜ: MUTLAKA ASCII ve MUTLAK YOL ***
outRoot   = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V2";      % örnek: C:\BALANCED_V2  (ASCII tut)

% Ayrım ve hedef büyüklük
valRatioPos    = 0.30;             % pozitiflerin %30'u val/test
targetPosTrain = 3000;             % augment sonrası TRAIN pozitif hedefi

% Ses ayarları
srTarget = 16000;

%% ===================== KAYNAKLARI OKU & ÇIKTI KLASÖRLERİ =====================
adsPos = audioDatastore(posDir,'IncludeSubfolders',true,'FileExtensions','.wav');
adsNeg = audioDatastore(negDir,'IncludeSubfolders',true,'FileExtensions','.wav');
nPos = numel(adsPos.Files); nNeg = numel(adsNeg.Files);
fprintf("Pozitif=%d, Negatif=%d\n", nPos, nNeg);
if nPos < 50, error("Pozitif çok az görünüyor. posDir'i kontrol edin."); end

% Alt yollar
outPosTr = fullfile(outRoot,"train","pos");
outPosVa = fullfile(outRoot,"val","pos");
outNegTr = fullfile(outRoot,"train","neg");
outNegVa = fullfile(outRoot,"val","neg");

% Klasörleri oluştur (fallback'li)
safe_mkdir(fileparts(outPosTr));  safe_mkdir(outPosTr);
safe_mkdir(fileparts(outPosVa));  safe_mkdir(outPosVa);
safe_mkdir(fileparts(outNegTr));  safe_mkdir(outNegTr);
safe_mkdir(fileparts(outNegVa));  safe_mkdir(outNegVa);

disp("=== ÇIKTI KLASÖRLERİ ===");
disp(outPosTr); disp(outPosVa); disp(outNegTr); disp(outNegVa);

%% ===================== 1) POZİTİF: %30 VAL/TEST AYIR (AUGMENT YOK) =====================
idx = randperm(nPos);
nVal = max(1, round(nPos*valRatioPos));
posValFiles  = string(adsPos.Files(idx(1:nVal)));
posTrainOrig = string(adsPos.Files(idx(nVal+1:end)));   % augment edilecek kaynaklar

for i=1:numel(posValFiles)
    [x,fs] = read_wav_mono(posValFiles(i), srTarget);
    seg    = one_sec_crop(x, fs);
    safe_write_wav(fullfile(outPosVa, sprintf("pos_val_%05d.wav", i)), seg, fs);
end
fprintf("VAL POS yazıldı: %d\n", numel(posValFiles));

%% ===================== 2) POZİTİF: TRAIN İÇİN KAYNAK + AUGMENT =====================
posTrainList = strings(0,1);

% a) Kaynaklardan direkt 1 sn crop (augment yok)
for i=1:numel(posTrainOrig)
    [x,fs] = read_wav_mono(posTrainOrig(i), srTarget);
    seg    = one_sec_crop(x, fs);
    fn = fullfile(outPosTr, sprintf("pos_src_%05d.wav", i));
    safe_write_wav(fn, seg, fs);
    posTrainList(end+1) = fn; %#ok<SAGROW>
end

% b) Augment ile hedefe tamamla (yalnızca time-stretch ±%8 ve pitch ±1..2 semiton)
augNeeded = max(0, targetPosTrain - numel(posTrainList));
fprintf("Augment üretilecek adet: %d (hedef TRAIN POS=%d)\n", augNeeded, targetPosTrain);

for k=1:augNeeded
    src = posTrainOrig( randi(numel(posTrainOrig)) );
    [x,fs] = read_wav_mono(src, srTarget);
    a = one_sec_crop(x, fs);

    % ±%8 time-stretch (0.92–1.08 aralığı)
    a = aug_timescale(a, 0.92 + 0.16*rand());

    % ±1..2 semiton pitch shift (yaklaşık yöntem)
    if rand < 0.80
        pick = [-2 -1 1 2];
        sem  = pick(randi(numel(pick)));
        a    = aug_pitch_shift_semitone(a, fs, sem);
    end

    % güvenlik normalize (soft)
    pk = max(abs(a)); if pk > 0.999, a = a / (pk + 1e-12) * 0.999; end

    fn = fullfile(outPosTr, sprintf("pos_aug_%05d.wav", k));
    safe_write_wav(fn, a, fs);
    posTrainList(end+1) = fn; %#ok<SAGROW>
end
fprintf("Toplam TRAIN POS: %d\n", numel(posTrainList));

%% ===================== 3) NEGATİF: VAL & TRAIN DENGELE =====================
% VAL NEG sayısı = VAL POS sayısı (dengeli validasyon)
nNegVal = numel(posValFiles);
negIdx  = randperm(nNeg);
negValFiles  = string(adsNeg.Files(negIdx(1:nNegVal)));
negTrainPool = string(adsNeg.Files(negIdx(nNegVal+1:end)));

% TRAIN NEG sayısı = TRAIN POS sayısı
needNegTrain = numel(posTrainList);
if numel(negTrainPool) < needNegTrain
    warning("Negatif train havuzu yetersiz (%d). Mevcut tümü kullanılacak.", numel(negTrainPool));
    needNegTrain = numel(negTrainPool);
end
negTrainSel = negTrainPool(randperm(numel(negTrainPool), needNegTrain));

% VAL NEG yaz
for i=1:numel(negValFiles)
    [x,fs] = read_wav_mono(negValFiles(i), srTarget);
    seg    = one_sec_crop(x, fs);
    safe_write_wav(fullfile(outNegVa, sprintf("neg_val_%05d.wav", i)), seg, fs);
end

% TRAIN NEG yaz
for i=1:numel(negTrainSel)
    [x,fs] = read_wav_mono(negTrainSel(i), srTarget);
    seg    = one_sec_crop(x, fs);
    safe_write_wav(fullfile(outNegTr, sprintf("neg_train_%06d.wav", i)), seg, fs);
end

fprintf("ÖZET → VAL POS: %d | VAL NEG: %d | TRAIN POS: %d | TRAIN NEG: %d\n", ...
    numel(posValFiles), numel(negValFiles), numel(posTrainList), numel(negTrainSel));

return; % ==== Local fonksiyonlar aşağıda olmalı (MATLAB kuralı) ====

%% ===================== LOCAL FONKSİYONLAR =====================
function ok = safe_mkdir(p)
    p = char(p);
    [ok,msg] = mkdir(p);
    if ~ok
        % Windows komutu ile dene
        [status,~] = system(['mkdir "' p '"']);
        if status ~= 0
            % Java fallback (Unicode destekli)
            f = java.io.File(p);
            ok = f.mkdirs();
            if ~ok
                error("mkdir başarısız: %s", msg);
            end
        else
            ok = true;
        end
    end
end

function [x, fs] = read_wav_mono(path, srTarget)
    [y, fs] = audioread(path);
    if size(y,2)>1, y = mean(y,2); end
    if fs ~= srTarget, y = resample(y, srTarget, fs); fs = srTarget; end
    pk = max(abs(y));
    if pk > 0, x = y / pk; else, x = y; end
end

function seg = one_sec_crop(x, fs)
    L = fs*1.0;
    if numel(x) < L
        seg = [x; zeros(L-numel(x),1)];
    else
        s0 = randi([1, numel(x)-L+1]);
        seg = x(s0:s0+L-1);
    end
end

function x2 = aug_timescale(x, factor) % 0.92–1.08
    L = numel(x);
    x2 = resample(x, round(1000*factor), 1000);
    if numel(x2) >= L, x2 = x2(1:L); else, x2(end+1:L) = 0; end
end

function x2 = aug_pitch_shift_semitone(x, fs, semitones) % -2..+2
    % Yaklaşık yöntem: resample ile pitch değiştir, uzunluğu geri düzelt
    factor = 2^(semitones/12);
    y = resample(x, round(1000*factor), 1000);
    x2 = resample(y, fs, fs);
    L  = fs; 
    if numel(x2) >= L, x2 = x2(1:L); else, x2(end+1:L) = 0; end
end

function safe_write_wav(path, x, fs)
    x(~isfinite(x)) = 0;
    mx = max(abs(x));
    if mx > 0.999
        x = x ./ (mx + 1e-12) * 0.999;  % soft limit (clipping engel)
    end
    audiowrite(path, x, fs);
end
