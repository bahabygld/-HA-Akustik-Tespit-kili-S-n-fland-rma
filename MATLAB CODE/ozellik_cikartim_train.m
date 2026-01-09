%% ====== FEATURE EXTRACTION: TRAIN ONLY (157D) + NOISE AUG ======
clear; clc; rng(42);  % tekrarlanabilirlik

% --- DÜZENLE (sende bu path mevcut) ---
rootOut = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V2";

% Ses & özellik parametreleri
srTarget = 16000;
winDur   = 0.025;   % 25 ms
hopDur   = 0.010;   % 10 ms
NUM_MFCC = 13;
NUM_MEL  = 64;      % log-Mel band sayısı

% --- Gürültü/augmentasyon ayarları ---
noise.enable         = true;                 % kapatmak için false
noise.type           = "awgn";               % "awgn" | "bg"
noise.includeClean   = true;                 % temiz örneği de dahil et
noise.snrList_dB     = [20 15 10 5];         % dB
noise.maxAugPerFile  = 1;                    % her snr için 1 kopya
noise.bgDir          = "";                   % "bg" modunda gürültü klasörü (.wav)
                                            % boşsa ve type="bg" ise awgn'a düşer

% ---- TRAIN ----
trainPosDir = fullfile(rootOut,"train","pos");
trainNegDir = fullfile(rootOut,"train","neg");

assert(isfolder(trainPosDir), "Train POS klasörü yok: %s", trainPosDir);
assert(isfolder(trainNegDir), "Train NEG klasörü yok: %s", trainNegDir);

fprintf("Train POS klasörü: %s\n", trainPosDir);
fprintf("Train NEG klasörü: %s\n", trainNegDir);

Tpos_tr = extract_features_from_folder(trainPosDir, 'pos', ...
    srTarget, NUM_MFCC, NUM_MEL, winDur, hopDur, noise);

Tneg_tr = extract_features_from_folder(trainNegDir, 'neg', ...
    srTarget, NUM_MFCC, NUM_MEL, winDur, hopDur, noise);

T_train = [Tpos_tr; Tneg_tr];
T_train.Split = categorical(repmat("train", height(T_train),1));

% Özet ve örnek satırlar
disp("Dağılım (train):");
disp(groupsummary(T_train, 'LabelCat'));
disp("İlk 5 satır:");
selCols = [1:6, width(T_train)-4:width(T_train)]; % son kolonlar meta
disp(T_train(1:min(5,height(T_train)), selCols));

% Kaydet
save(fullfile(rootOut,'T_train.mat'), 'T_train', '-v7.3');
disp("Kaydedildi: " + fullfile(rootOut,'T_train.mat'));

%% ===================== LOCAL FUNCTIONS (DOSYANIN SONUNDA OLMALI) =====================

function [x, fs] = read_wav_mono(path, srTarget)
    [y, fs] = audioread(path);
    if size(y,2)>1, y = mean(y,2); end
    if fs ~= srTarget, y = resample(y, srTarget, fs); fs = srTarget; end
    pk = max(abs(y)); if pk>0, x = y/pk; else, x = y; end
end

function v = fixnan(v)
    v(~isfinite(v)) = 0;
end

function v = ensure_len(v,n)
    v = v(:).';
    if numel(v) < n
        v(end+1:n) = 0;
    elseif numel(v) > n
        v = v(1:n);
    end
end

function [mf_m,mf_s] = safe_mfcc(x, fs, K)
    C = mfcc(x,fs,'NumCoeffs',K);          % frames x K
    if size(C,1) < size(C,2), C = C.'; end % uyumluluk
    mf_m = ensure_len(fixnan(mean(C,1)),K);
    mf_s = ensure_len(fixnan(std (C,0,1)),K);
end

function [ml_m,ml_s] = safe_logmel(x, fs, B, winDur, hopDur)
    winLen = max(32, round(winDur*fs));
    hopLen = max(16, round(hopDur*fs));
    win    = hamming(winLen, 'periodic');

    S = melSpectrogram(x, fs, ...
        'Window',        win, ...
        'OverlapLength', winLen - hopLen, ...
        'NumBands',      B, ...
        'FFTLength',     1024);             % B x frames

    Slog = log10(S + 1e-12);
    ml_m = ensure_len(fixnan(mean(Slog,2).'), B);
    ml_s = ensure_len(fixnan(std (Slog,0,2).'), B);
end

function x_noisy = add_awgn_to_snr(x, targetSNRdB)
    % x: [-1,1] normalize
    Px = mean(x.^2) + eps;
    targetSNR = 10^(targetSNRdB/10);
    Pn = Px/targetSNR;
    n = sqrt(Pn) * randn(size(x));
    x_noisy = x + n;
    % yeniden normalize etmiyoruz; SNR korunur
end

function x_noisy = mix_with_bg_to_snr(x, bg, targetSNRdB)
    % bg uzunluğu x'ten kısa/uzunsa döngüyle eşle
    if numel(bg) < numel(x)
        rep = ceil(numel(x)/numel(bg));
        bg  = repmat(bg, rep, 1);
    end
    bg = bg(1:numel(x));

    Px = mean(x.^2) + eps;
    Pb = mean(bg.^2) + eps;

    % bg'yi SNR hedefine göre ölçekle
    alpha = sqrt(Px / (Pb * 10^(targetSNRdB/10)));
    x_noisy = x + alpha*bg;
end

function bg = load_random_bg(bgDir, srTarget)
    persistent bgAds
    if isempty(bgAds)
        if strlength(bgDir)>0 && isfolder(bgDir)
            bgAds = audioDatastore(bgDir, 'IncludeSubfolders', true, 'FileExtensions','.wav');
        else
            bgAds = [];
        end
    end
    if isempty(bgAds) || isempty(bgAds.Files)
        bg = []; return;
    end
    idx = randi(numel(bgAds.Files));
    [b,fs] = audioread(bgAds.Files{idx});
    if size(b,2)>1, b = mean(b,2); end
    if fs ~= srTarget, b = resample(b, srTarget, fs); end
    % pk normalize etmiyoruz; enerji oranına göre zaten ölçeklenecek
    bg = b;
end

function T = extract_features_from_folder(rootDir, labelName, ...
    srTarget, NUM_MFCC, NUM_MEL, winDur, hopDur, noise)

    ads = audioDatastore(rootDir,'IncludeSubfolders',true,'FileExtensions','.wav');
    feats = {}; labs = {}; files = {}; ntypes = {}; snrs = []; aidxs = [];
    n = numel(ads.Files);

    useBG = (noise.enable && noise.type=="bg" && strlength(noise.bgDir)>0 && isfolder(noise.bgDir));

    for i = 1:n
        src = ads.Files{i};
        try
            [x,fs] = audioread(src);
            if size(x,2)>1, x = mean(x,2); end
            if fs ~= srTarget, x = resample(x, srTarget, fs); fs = srTarget; end
            pk = max(abs(x)); if pk>0, x = x/pk; end

            % === VARYASYON SETİ ===
            varX = {};
            varNoiseType = strings(0);
            varSNR = [];
            varAugIdx = [];

            % 0) temiz
            if ~noise.enable || noise.includeClean
                varX{end+1}        = x;
                varNoiseType(end+1)= "clean";
                varSNR(end+1)      = NaN;
                varAugIdx(end+1)   = 0;
            end

            if noise.enable
                for s = 1:numel(noise.snrList_dB)
                    snrDB = noise.snrList_dB(s);
                    for rep = 1:noise.maxAugPerFile
                        if useBG
                            bg  = load_random_bg(noise.bgDir, srTarget);
                            if isempty(bg)
                                xaug = add_awgn_to_snr(x, snrDB); 
                                ntype = "awgn"; % fallback
                            else
                                xaug = mix_with_bg_to_snr(x, bg, snrDB);
                                ntype = "bg";
                            end
                        else
                            if noise.type=="bg"
                                % bg klasörü yoksa awgn'a düş
                                xaug = add_awgn_to_snr(x, snrDB);
                                ntype = "awgn";
                            else
                                xaug = add_awgn_to_snr(x, snrDB);
                                ntype = "awgn";
                            end
                        end
                        % hafif clip öncesi sınırla (isteğe bağlı)
                        xaug = max(min(xaug, 1), -1);

                        varX{end+1}        = xaug;
                        varNoiseType(end+1)= ntype;
                        varSNR(end+1)      = snrDB;
                        varAugIdx(end+1)   = rep;
                    end
                end
            end

            % === TÜM VARYASYONLAR İÇİN ÖZELLİK ÇIKARIMI ===
            for k = 1:numel(varX)
                xx = varX{k};

                % --- MFCC (26) ---
                [mf_m,mf_s] = safe_mfcc(xx,fs,NUM_MFCC);

                % --- log-Mel (128) ---
                [ml_m,ml_s] = safe_logmel(xx,fs,NUM_MEL,winDur,hopDur);

                % --- Diğerleri (3) ---
                zcrVal = mean(zerocrossrate(xx));
                rmsVal = rms(xx);
                scVal  = mean(spectralCentroid(xx,fs));

                feat = [mf_m, mf_s, ml_m, ml_s, zcrVal, rmsVal, scVal]; % 157D
                feats{end+1,1} = single(feat);
                labs {end+1,1} = string(labelName);
                files{end+1,1} = string(src);
                ntypes{end+1,1}= varNoiseType(k);
                snrs (end+1,1) = varSNR(k);
                aidxs(end+1,1) = varAugIdx(k);
            end

        catch ME
            warning("Atlandı: %s -> %s", src, ME.message);
        end

        if mod(i, max(1,floor(n/20)))==0
            fprintf("İlerleme (%s): %d/%d\n", labelName, i, n);
        end
    end

    X = vertcat(feats{:});
    fn = "F_" + (1:size(X,2));
    T  = array2table(X, 'VariableNames', fn);

    labStr       = vertcat(labs{:});
    T.LabelCat   = categorical(labStr);
    T.Label01    = double(T.LabelCat=="pos");
    T.File       = vertcat(files{:});

    % augmentasyon meta
    T.NoiseType  = categorical(string(vertcat(ntypes{:})));
    T.SNR        = snrs;        % dB (clean -> NaN)
    T.AugIdx     = aidxs;       % 0=clean, >=1 augment tekrarı
end
