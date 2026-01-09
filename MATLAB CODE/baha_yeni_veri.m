%% === 1) Dataset & sanity checks ===
clear; clc;

datasetPath = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\İha Akustik ses\Veri_Setleri\yeni"; % içinde: 0 ve 1 (veya pozitif klasör)
ads = audioDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.wav', ...
    'LabelSource', 'foldernames');

numFiles = numel(ads.Files);
fprintf('%d dosya bulundu.\n', numFiles);

% Hangi etiketler var ve kaç tane?
if numFiles==0
    error('Hiç WAV bulunamadı. datasetPath veya alt klasörleri kontrol et.');
end
disp('Bulunan sınıflar ve adetleri:');
disp(countEachLabel(ads));

% Etiket isimlerini net görelim
uniqLabs = categories(ads.Labels);
disp('Etiket isimleri (foldernames):'); disp(uniqLabs);

% Eğer beklediğin "0" ve "1" yoksa uyar:
if ~any(strcmpi(uniqLabs,'0'))
    warning('Uyarı: "0" etiketi bulunamadı. Negatif sınıf klasör adı gerçekten "0" mı?');
end
% Pozitif klasör adı "1" değilse sorun değil; aşağıda tüm 0-dışı "pos" sayılacak.

%% === 2) Parametreler ===
winDur   = 0.025;   % 25 ms
hopDur   = 0.010;   % 10 ms
NUM_MFCC = 13;
NUM_MEL  = 64;      % log-Mel bant sayısı

%% === 3) Özellik çıkarımı (157D) ===
featureCells = {};
labelCells   = {};
fileCells    = {};   % <-- işlenen gerçek dosya yolu burada tutulacak
okCount = 0;

reset(ads);  % güvenli başlangıç
for i = 1:numFiles
    src = ads.Files{i};
    try
        [x, fs] = audioread(src);
        if size(x,2) > 1, x = mean(x,2); end
        mx = max(abs(x)); if mx > 0, x = x/mx; end

        % Çok kısa sinyalleri min 40 ms'e pad et
        minLen = max(1, round(0.04*fs));
        if numel(x) < minLen, x(end+1:minLen) = 0; end

        % MFCC (13 mean + 13 std = 26)
        [mfccMean, mfccStd] = safe_mfcc(x, fs, NUM_MFCC);

        % log-Mel (64 mean + 64 std = 128)
        [melMean, melStd] = safe_logmel(x, fs, NUM_MEL, winDur, hopDur);

        % ZCR / RMS / Spectral Centroid (3)
        zcrVal = mean(zerocrossrate(x));
        rmsVal = rms(x);
        sc     = spectralCentroid(x, fs);
        scMean = mean(sc);

        % Özellik vektörü (26 + 128 + 3 = 157)
        feat = [mfccMean, mfccStd, melMean, melStd, zcrVal, rmsVal, scMean];
        feat = ensure_len(fixnan(feat), 157);

        featureCells{end+1,1} = single(feat); %#ok<SAGROW>
        labelCells  {end+1,1} = ads.Labels(i); %#ok<SAGROW>
        fileCells   {end+1,1} = string(src);   %#ok<SAGROW>
        okCount = okCount + 1;

    catch ME
        warning('Atlandı: %s -> %s', src, ME.message);
    end

    if mod(i, max(1, floor(numFiles/20))) == 0
        fprintf('İlerleme: %d/%d\n', i, numFiles);
    end
end

if okCount==0
    error('Hiç özellik çıkarılamadı.');
end

X       = vertcat(featureCells{:});                 % N x 157
labsRaw = categorical(vertcat(labelCells{:}));

%% === 4) Etiket eşleme (yalnızca '0' => neg/0, diğerlerinin hepsi => pos/1) ===
labsStr = lower(string(labsRaw));
% doygun normalize (alt çizgi, boşluk vs. kaldırmak isteyebilirsin; gerekmez)
Label01  = double(labsStr ~= "0");                 % '0' -> 0, kalan her şey -> 1
LabelCat = categorical(Label01, [0 1], {'neg','pos'});

%% === 5) Tablo + Kaydet ===
fn = "F_" + (1:size(X,2));
T = array2table(X, 'VariableNames', fn);
T.LabelCat = LabelCat;
T.Label01  = Label01;
T.File     = fileCells;    % <-- işlenen gerçek dosya yolları

% Özet yazdır
nNeg = sum(T.Label01==0);
nPos = sum(T.Label01==1);
fprintf('Toplam: %d örnek (neg: %d, pos: %d)\n', height(T), nNeg, nPos);

% Sınıf örneklerinden hızlı örnek gösterelim
disp('--- Neg örneklerinden ilk 3 ---');
disp(T(find(T.Label01==0, 3, 'first'), [1:6 end-2:end]));
disp('--- Pos örneklerinden ilk 3 ---');
if nPos>0
    disp(T(find(T.Label01==1, 3, 'first'), [1:6 end-2:end]));
else
    warning('Pozitif örnek bulunamadı! Pozitif klasör adı/konumu yanlış olabilir.');
end

save('audioFeatures.mat', 'T');
fprintf('audioFeatures.mat kaydedildi. Classification Learner için hazır.\n');

return; % === Script burada biter; aşağıda local fonksiyonlar ===

%% ======================= LOCAL FONKSİYONLAR =======================
function v = fixnan(v)
    v(~isfinite(v)) = 0;
end

function v = ensure_len(v, n)
    v = v(:).';
    if numel(v) < n
        v(end+1:n) = 0;
    elseif numel(v) > n
        v = v(1:n);
    end
end

function [mfccMean, mfccStd] = safe_mfcc(x, fs, NUM_MFCC)
    C = mfcc(x, fs, 'NumCoeffs', NUM_MFCC);
    if size(C,1) < size(C,2), C = C.'; end
    m = mean(C, 1);
    s = std (C, 0, 1);
    mfccMean = ensure_len(fixnan(m(:).'), NUM_MFCC);
    mfccStd  = ensure_len(fixnan(s(:).'), NUM_MFCC);
end

function [melMean, melStd] = safe_logmel(x, fs, NUM_MEL, winDur, hopDur)
    winLen = max(32, round(winDur*fs));
    hopLen = max(16, round(hopDur*fs));
    win    = hamming(winLen, 'periodic');
    S = melSpectrogram(x, fs, ...
        'Window',        win, ...
        'OverlapLength', winLen - hopLen, ...
        'NumBands',      NUM_MEL, ...
        'FFTLength',     1024);           % [NUM_MEL x frames]
    Slog = log10(S + 1e-12);
    melMean = ensure_len(fixnan(mean(Slog, 2).'), NUM_MEL);
    melStd  = ensure_len(fixnan(std (Slog, 0, 2).'), NUM_MEL);
end
