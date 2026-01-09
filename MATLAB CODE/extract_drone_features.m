function feat = extract_drone_features(wavFile)
% EXTRACT_DRONE_FEATURES
%   Tek bir .wav dosyasi icin, egitimde kullandiginla
%   birebir AYNI 157 boyutlu ozellik vektorunu uretir.
%
%   girdi:
%       wavFile : tam dosya yolu (string veya char)
%
%   cikti:
%       feat    : 1 x 157 single vektor

    % --- Egitimde kullandigin sabitler ---
    srTarget = 16000;
    winDur   = 0.025;
    hopDur   = 0.010;
    NUM_MFCC = 13;
    NUM_MEL  = 64;

    % 1) Ses dosyasini oku
    [x, fs] = audioread(wavFile);

    % 2) Mono'ya cevir
    if size(x,2) > 1
        x = mean(x, 2);
    end

    % 3) Hedef orneleme frekansina getir (16 kHz)
    if fs ~= srTarget
        x = resample(x, srTarget, fs);
        fs = srTarget;
    end

    % 4) Birim tepe (unit peak) normalizasyonu
    x = unit_peak(x);

    % 5) Egitimde kullandigin 157D feature fonksiyonu
    feat = extract_157d(x, fs, NUM_MFCC, NUM_MEL, winDur, hopDur);

    % vectoru 1 x D hale getir (emin olmak icin)
    feat = single(feat(:)).';
end

% ================== LOCAL FUNCTIONS (egitimdekiyle AYNI) ==================

function x = unit_peak(x)
    pk = max(abs(x));
    if pk > 0
        x = x / pk;
    end
end

function feat = extract_157d(x, fs, NUM_MFCC, NUM_MEL, winDur, hopDur)
    % ---- MFCC ----
    C = mfcc(x, fs, 'NumCoeffs', NUM_MFCC);
    if size(C,1) < size(C,2)
        C = C.';
    end
    mf_m = mean(C, 1);
    mf_s = std(C, 0, 1);

    % ---- Log-Mel Spektrogram ----
    wl  = max(32, round(winDur * fs));
    hl  = max(16, round(hopDur * fs));
    win = hamming(wl, 'periodic');

    S = melSpectrogram(x, fs, ...
        'Window',        win, ...
        'OverlapLength', wl - hl, ...
        'NumBands',      NUM_MEL, ...
        'FFTLength',     1024);

    Slog = log10(S + 1e-12);
    ml_m = mean(Slog, 2).';
    ml_s = std(Slog, 0, 2).';

    % ---- Ek akustik ozellikler ----
    zcr = mean(zerocrossrate(x));
    rv  = rms(x);
    sc  = mean(spectralCentroid(x, fs));

    % ---- Hepsini birlestir (toplam 157 boyut) ----
    feat = [mf_m, mf_s, ml_m, ml_s, zcr, rv, sc];
end
