function iha_zero_split_1000()
% Sadece 0 klasörünü işler; 1 sn segmentler üretir ve
% ardışık her 1000 segmenti tek bir grup olarak etiketler.
%
% Çıktılar:
%   out_zero/0_segments/*.wav
%   out_zero/zero_manifest.csv  (outfile, orig_id, seg_idx, seg_start_s, grp_id)

%% === AYARLAR ===
rootDir = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\İha Akustik ses\Veri_Setleri\veri_seti_dads";
dir0    = fullfile(rootDir,"0");
outDir  = fullfile(rootDir,"out_zero");
outWav  = fullfile(outDir,"0_segments");   % sadece 0 sınıfı segmentleri
FsT     = 16000;
segSec  = 1.0;
win     = round(segSec*FsT);
rng(42);

if ~exist(outWav,'dir'), mkdir(outWav); end

% Manifest tablosu (boş, doğru tiplerle)
Man = table(strings(0,1), strings(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    'VariableNames', {'outfile','orig_id','seg_idx','seg_start_s','grp_id'});

%% === VERİYİ YÜKLE ===
ads0 = audioDatastore(dir0,'IncludeSubfolders',true,'FileExtensions',{'.wav'});
files0 = ads0.Files;
fprintf('[0] toplam dosya: %d\n', numel(files0));

%% === İŞLEME ===
tStart = tic;
totalSeg = 0;

for i = 1:numel(files0)
    f = files0{i};
    [~,base,~] = fileparts(f);

    % güvenli okuma
    try
        [x,Fs] = audioread(f);
    catch ME
        warning('[SKIP] okuma hatası: %s | %s', f, ME.message);
        continue
    end

    % mono + hedef örnekleme
    if size(x,2)>1, x = mean(x,2); end
    if Fs ~= FsT, x = resample(x, FsT, Fs); end
    x = x ./ max(1e-9, max(abs(x)));

    L = numel(x);
    nSeg = floor(L/win);
    if nSeg < 1
        if mod(i,200)==0 || i==numel(files0)
            fprintf('[0] %d/%d | totalSeg=%d | t=%.1fs\n', i, numel(files0), totalSeg, toc(tStart));
        end
        continue
    end

    % segmentle ve yaz
    for k = 1:nSeg
        seg = x((k-1)*win+1 : k*win);

        % global segment index → grup id (1000’lik paketler)
        globalIdx = totalSeg;              % 0-based sayacağız
        grp_id    = floor(globalIdx / 1000);

        outName = sprintf('0_%s_%04d.wav', base, k);
        outPath = fullfile(outWav, outName);
        try
            audiowrite(outPath, seg, FsT);
        catch ME
            warning('[SKIP] yazma hatası: %s | %s', outPath, ME.message);
            continue
        end

        % manifest satırı
        Man = [Man; {string(outPath), string(base), k, (k-1)*segSec, grp_id}]; %#ok<AGROW>
        totalSeg = totalSeg + 1;
    end

    if mod(i,200)==0 || i==numel(files0)
        fprintf('[0-WRITE] %d/%d | totalSeg=%d | last grp=%d | t=%.1fs\n', ...
            i, numel(files0), totalSeg, floor((totalSeg-1)/1000), toc(tStart));
        drawnow limitrate
    end
end

%% === MANIFEST YAZ ===
outCsv = fullfile(outDir, 'zero_manifest.csv');
writetable(Man, outCsv);
fprintf('\nBİTTİ ✅  Toplam segment: %d | Grup sayısı (1000’lik): %d\n', totalSeg, floor((totalSeg+999)/1000));
fprintf('WAV klasörü : %s\nManifest     : %s\n', outWav, outCsv);
end
