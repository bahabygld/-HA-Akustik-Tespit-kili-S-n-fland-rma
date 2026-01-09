function iha_dosya_tanitma()
% HAFİF ama SAĞLAM clip-level özellik çıkarımı (paralel, 12 çekirdek, dengeli alt-küme)
% Özellikler (158D, opsiyonel 184D):
%  - MFCC mean+std (26), log-Mel mean+std (128), ZCR/RMS/Centroid (3), [opsiyonel] Δ MFCC mean+std (+26)
% Dengeleme: label==0 sayısı kadar (≈16,729) label==1 rastgele indirgenir (1:1).
% Çıktılar:
%   - T_clip_light (balanced, file, source_id, label sütunlarıyla)
%   - (ops.) T_train_bal / T_test_bal (%80/20 stratified hold-out)
%   - params, selected_indices.mat
%
% KAYIT: outMat değişkeninde belirtilen .mat dosyasına kaydedilir.

%% === KULLANICI AYARLARI ===
dataRoot   = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\İha Akustik ses\Veri_Setleri\veri_seti_dads";   % 0\ , 1\
outDir     = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\İha Akustik ses\Veri_Setleri\ozellik_cikarimi";
outMat     = fullfile(outDir, "dads_features_light.mat");

USE_DELTAS     = false;    % true: Δ MFCC mean+std ekler (+26D)
CLEAN_OLD      = true;     % outDir’deki eski *features*.mat dosyalarını sil
MAKE_HOLDOUT   = true;     % sabit %80/20 hold-out (T_train_bal, T_test_bal) oluştur
PARPOOL_WORKER = 12;       % paralel işçi sayısı

% Grup-bazlı CV için grup içi sınıf dengesi parametreleri
Kfold               = 5;   % ileride GroupKFold için uygun kat sayısı
grpSizePerClass     = 80;  % her grupta her sınıftan ~80 örnek (64–128 iyi aralık)

%% === Sabit parametreler (rapor için de saklanır) ===
fsTarget = 16000;       % 0–8 kHz bandı için yeterli, hızlı
winDur   = 0.025;       % 25 ms
hopDur   = 0.010;       % 10 ms
mfccNum  = 13;          % klasik kompakt temsil
nMels    = 64;          % 0–8 kHz’i dengeli örnekler

if ~exist(outDir,'dir'), mkdir(outDir); end
if CLEAN_OLD, delete(fullfile(outDir, "*features*.mat")); end

rng(42,'twister');      % tekrar üretilebilirlik

%% === Datastore ===
ads = audioDatastore(dataRoot,'IncludeSubfolders',true, ...
    'FileExtensions','.wav','LabelSource','foldernames');

filesFull  = ads.Files;
labelsFull = categorical(ads.Labels);
Nfull = numel(filesFull);
assert(Nfull>0, "WAV bulunamadı: %s", dataRoot);
fprintf('Toplam dosya: %d\n', Nfull);

% %% === Sınıf dengeleme (1:1) — label==1 downsample to label==0 count (≈16,729) ===
% idx0 = find(labelsFull==categorical(0));
% idx1 = find(labelsFull==categorical(1));
%  %#ok<CTGUS>  % AUTOCORR altı çizmesin diye
% % düzeltme:
% idx1 = find(labelsFull==categorical(1));
% assert(~isempty(idx0) && ~isempty(idx1), 'Sınıf indeksleri boş.');
% 
% targetN = numel(idx0);                    % 0 sınıf sayısı (≈ 16,729)
% if numel(idx1) < targetN
%     error('label==1 sayısı (%d) targetN (%d) den küçük; bu senaryo beklenmedi.', numel(idx1), targetN);
% end
% 
% idx0r = idx0;                             % 0'lar olduğu gibi (istersen buradan da kısabilirsin)
% idx1r = randsample(idx1, targetN);        % 1'leri 16,729’a indir
% 
% sel = sort([idx0r; idx1r]);               % dengeli alt-küme
% files  = filesFull(sel);
% labels = labelsFull(sel);
% N = numel(sel);

%% === Dengeleme YOK — TÜM VERİ KULLANILIR ===
files  = filesFull;
labels = labelsFull;
N = numel(files);

fprintf('Dengeleme yapılmadı. Toplam veri: %d | label==0: %d | label==1: %d\n', ...
    N, sum(labels==categorical(0)), sum(labels==categorical(1)));


fprintf('Denge sonrası -> label==0: %d, label==1: %d (Toplam=%d)\n', ...
    sum(labels==categorical(0)), sum(labels==categorical(1)), N);

% Reprodüksiyon için seçim bilgisini kaydet
selectedInfo = struct('selIdx', sel, 'targetN', targetN, 'rngSeed', 42);
save(fullfile(outDir,'selected_indices.mat'), 'selectedInfo');

%% === Çerçeve parametreleri ===
winLen = round(winDur*fsTarget);
hopLen = round(hopDur*fsTarget);
win    = hamming(winLen,'periodic');

%% === Özellik boyutu (hafif) ===
D = 26 + 128 + 3;          % MFCC(26) + logMel(128) + ZCR/RMS/Cent(3) = 158
if USE_DELTAS, D = D + 26; end
X = zeros(N, D, 'single'); % preallocate

%% === Paralel havuz (12 çekirdek) ===
pool = gcp('nocreate');
if isempty(pool)
    parpool("local", PARPOOL_WORKER);
else
    fprintf('Paralel havuz zaten açık: %d worker\n', pool.NumWorkers);
end

%% === Progress (DataQueue) ===
dq = parallel.pool.DataQueue;
pCount = 0; t0 = tic;
afterEach(dq, @prog);

%% === Ana döngü (parfor) ===
parfor i = 1:N
    try
        [x,fs] = audioread(files{i});
        X(i,:) = computeClipFeatures_par( ...
            x, fs, fsTarget, win, hopLen, winLen, mfccNum, nMels, USE_DELTAS );
    catch
        X(i,:) = zeros(1,D,'single');  % problemli dosyaya karşı güvenlik
    end
    send(dq, 1);
end

%% === Tablo + Kaydet ===
featNames = "F_" + string(1:D);
T_clip_light = array2table(X, 'VariableNames', featNames);

% === [YENİ] Dosya yolu + DENGELİ GRUP KİMLİĞİ (source_id) ===
% Amaç: GroupKFold'da her fold'da her iki sınıftan da örnek olsun, tek-sınıflı fold olmasın.

% 1) Dosya yolu ve etiket
T_clip_light.file  = string(files);
T_clip_light.label = categorical(labels);

% 2) Sınıf indeksleri
y = T_clip_light.label;
idx0 = find(y==categorical(0));
idx1 = find(y==categorical(1));

% 3) Aynı grupta her iki sınıftan örnek olacak şekilde karıştır + dilimle
rng(42,'twister');
idx0 = idx0(randperm(numel(idx0)));
idx1 = idx1(randperm(numel(idx1)));

g0 = ceil((1:numel(idx0))'/grpSizePerClass);
g1 = ceil((1:numel(idx1))'/grpSizePerClass);
m  = max(max(g0), max(g1));     % ortak grup sayısı

% Grup sayısını Kfold'un katına yuvarla (stratified GroupKFold kolaylaşsın)
r = mod(m,Kfold);
if r~=0, m = m + (Kfold - r); end

G = strings(height(T_clip_light),1);
for g = 1:m
    name = "g"+string(g);

    a0 = (g-1)*grpSizePerClass + 1;  b0 = min(g*grpSizePerClass, numel(idx0));
    if a0<=b0, G(idx0(a0:b0)) = name; end

    a1 = (g-1)*grpSizePerClass + 1;  b1 = min(g*grpSizePerClass, numel(idx1));
    if a1<=b1, G(idx1(a1:b1)) = name; end
end

% Güvenlik: boş kalan örnek varsa g1'e ata
empty = (G=="");
if any(empty), warning('Boş group ataması: %d adet. g1''e taşınıyor.', sum(empty)); G(empty) = "g1"; end

T_clip_light.source_id = G;   % <<< group-based CV anahtarı

% Rapor parametreleri
params = struct('dataRoot',string(dataRoot),'fsTarget',fsTarget, ...
                'winDur',winDur,'hopDur',hopDur, ...
                'mfccNum',mfccNum,'nMels',nMels,'useDeltas',USE_DELTAS, ...
                'balanceMethod',"downsample-1to0",'targetN',targetN, ...
                'grpSizePerClass',grpSizePerClass,'Kfold',Kfold, ...
                'rngSeed',42);

% Kayıt
if MAKE_HOLDOUT
    % stratified hold-out (label'a göre)
    cv = cvpartition(T_clip_light.label,'HoldOut',0.20);
    T_train_bal = T_clip_light(training(cv),:);
    T_test_bal  = T_clip_light(test(cv),:);
    save(outMat, 'T_clip_light','T_train_bal','T_test_bal','params','-v7.3');
    fprintf('Kaydedildi: %s  | T_clip_light: %d x %d | Hold-out: train=%d, test=%d\n', ...
        outMat, height(T_clip_light), width(T_clip_light), height(T_train_bal), height(T_test_bal));
else
    save(outMat, 'T_clip_light','params','-v7.3');
    fprintf('Kaydedildi: %s  | T_clip_light: %d x %d\n', ...
        outMat, height(T_clip_light), width(T_clip_light));
end

% (Ops.) delete(gcp('nocreate'));

%% === iç: progress yazdırıcı ===
    function prog(~)
        pCount = pCount + 1;
        f  = pCount / N; el = toc(t0);
        eta = el*(1-f)/max(f,eps);
        if mod(pCount, max(1,round(N/100)))==0 || pCount==N
            fprintf('[%d/%d] Geçen: %s | Kalan ~%s\n', ...
                pCount, N, duration(0,0,el,"Format","hh:mm:ss"), duration(0,0,eta,"Format","hh:mm:ss"));
        end
    end
end

% ====== ALT FONKSİYON (parfor güvenli) ======
function v = computeClipFeatures_par(x, fs, fsTarget, win, hopLen, winLen, mfccNum, nMels, USE_DELTAS)
    % Mono + resample + normalize
    if size(x,2)>1, x = mean(x,2); end
    if fs~=fsTarget, x = resample(x, fsTarget, fs); fs = fsTarget; end
    mx = max(abs(x)); if mx>0, x = x./mx; end

    % MFCC (frames x mfccNum)
    C = mfcc(x, fs, 'NumCoeffs', mfccNum, ...
             'Window', win, 'OverlapLength', numel(win)-hopLen, ...
             'LogEnergy','Ignore');
    mfccMean = mean(C,1); mfccStd  = std(C,0,1);
    extra = [];
    if USE_DELTAS
        CD    = diff([C(1,:); C],1,1);   % basit delta
        extra = [mean(CD,1), std(CD,0,1)];
    end

    % log-Mel (nMels x frames) -> dB
    S = melSpectrogram(x, fs, 'Window', win, ...
                       'OverlapLength', numel(win)-hopLen, ...
                       'NumBands', nMels, ...
                       'FFTLength', 2^nextpow2(numel(win)));
    Sdb = pow2db(S + eps);
    lmelMean = mean(Sdb, 2).';  lmelStd = std(Sdb, 0, 2).';

    % ZCR / RMS / Spectral Centroid
    z  = zerocrossrate(x, 'WindowLength', winLen, 'OverlapLength', winLen-hopLen);  zcrVal = mean(z);
    sc = spectralCentroid(x, fs, 'Window', win, 'OverlapLength', numel(win)-hopLen); specCent = mean(sc);
    eng = rms(x);

    % Birleştir
    v = single([mfccMean, mfccStd, extra, lmelMean, lmelStd, zcrVal, eng, specCent]);
    v(~isfinite(v)) = 0;
end
