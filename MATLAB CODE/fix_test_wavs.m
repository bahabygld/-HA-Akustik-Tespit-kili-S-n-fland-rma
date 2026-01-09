%% ===================== TEST WAV DÜZELTME (1 saniyeye pad/crop) =====================
clear; clc;

srcRoot = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V3_SON_TEST";
dstRoot = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V3_SON_TEST_FIXED";

srTarget = 16000;
L = srTarget * 1;   % 1 saniye = 16000 örnek

folders = ["0", "1"];

for f = folders
    srcDir = fullfile(srcRoot, f);
    dstDir = fullfile(dstRoot, f);

    if ~exist(dstDir,"dir")
        mkdir(dstDir);
    end

    files = dir(fullfile(srcDir,"*.wav"));
    fprintf("\nKlasör: %s  (%d dosya)\n", srcDir, numel(files));

    for i = 1:numel(files)
        inFile  = fullfile(files(i).folder, files(i).name);
        outFile = fullfile(dstDir, files(i).name);

        % WAV oku
        [x,fs] = audioread(inFile);
        if size(x,2)>1, x = mean(x,2); end

        % Resample → 16kHz
        if fs ~= srTarget
            x = resample(x, srTarget, fs);
        end

        % Normalize
        mx = max(abs(x));
        if mx>0, x = x / mx; end

        % 1 saniye crop/pad
        if numel(x) < L
            x = [x; zeros(L - numel(x), 1)];
        elseif numel(x) > L
            x = x(1:L);
        end

        % Kaydet
        audiowrite(outFile, x, srTarget);

        if mod(i, 500) == 0
            fprintf("%d / %d düzeltildi...\n", i, numel(files));
        end
    end
end

fprintf("\nTÜM WAV’LAR 1 SANIYE FORMATINA DÖNÜŞTÜRÜLDÜ!\n");
fprintf("Yeni test klasörü: %s\n", dstRoot);
