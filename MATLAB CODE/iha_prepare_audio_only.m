function iha_prepare_audio_only()
% === KULLANICI AYARLARI ===
rootDir = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\İha Akustik ses\Veri_Setleri\veri_seti_dads";
dir0   = fullfile(rootDir,"0");
dir1   = fullfile(rootDir,"1");
outDir = fullfile(rootDir,"out_audio");  % üretim klasörü
FsT    = 16000;
segSec = 1.0;                 % hedef segment süresi
win    = round(segSec*FsT);
rng(42)

% Class-1 grup belirleme modu:
%  "fast_numeric"  -> dosya adındaki 7 haneli sayıya göre ~1000'lik blok varsayımı (HIZLI)
%  "audio_similarity" -> ardışık log-Mel benzerliği ile grup (YAVAŞ - bu betikte KAPALI)
groupMode = "fast_numeric";

% --- ÇIKIŞ klasörlerini hazırla
folders = ["train","val","test"];
for f = folders
    for c = ["0","1"]
        tgt = fullfile(outDir, f, c);
        if ~exist(tgt,'dir'), mkdir(tgt); end
    end
end

fprintf('=== 0 SINIFI: 1 sn segmentlere bölünüyor ===\n')
ads0 = audioDatastore(dir0,'IncludeSubfolders',true, ...
    'FileExtensions',{'.wav','.flac','.mp3','.m4a','.ogg'});

% Class-0 manifest (boş doğru tiplerle)
M0 = table(strings(0,1), strings(0,1), strings(0,1), strings(0,1), zeros(0,1), ...
    'VariableNames', {'split','outfile','label','orig_id','seg_start_s'});

% Class-0: önce tüm dosyalardan kaç segment çıkacağını hesaplayıp
% RAM'e ses almadan, sadece (kaynak yol, segment index) tutuyoruz.
tStart = tic;
total0 = numel(ads0.Files);
cntSeg0 = 0;

tmpList0 = struct('orig_id',{},'parts',{});
for i = 1:total0
    f = ads0.Files{i};
    [~,base,~] = fileparts(f);
    try
        [x,Fs] = audioread(f);
    catch
        warning('Okuma hatası (0): %s', f); 
        continue
    end
    if size(x,2)>1, x = mean(x,2); end
    if Fs~=FsT, x = resample(x,FsT,Fs); end
    x = x ./ max(1e-9, max(abs(x)));
    L = numel(x);
    nSeg = floor(L/win);
    if nSeg<1
        if mod(i,200)==0 || i==total0
            fprintf('[0] %d/%d | seg:%d | t=%.1fs\n', i,total0,cntSeg0,toc(tStart));
        end
        continue
    end

    parts = cell(nSeg,1);
    for k=1:nSeg
        parts{k} = struct('fpath', f, 'k', k);  % sadece meta
    end
    tmpList0(end+1).orig_id = string(base); %#ok<AGROW>
    tmpList0(end).parts     = parts;        %#ok<AGROW>

    cntSeg0 = cntSeg0 + nSeg;
    if mod(i,200)==0 || i==total0
        fprintf('[0] %d/%d | üretilen segment adayları:%d | t=%.1fs\n', i,total0,cntSeg0,toc(tStart));
    end
end

% Class-0 grup id: her kaynak dosya ayrı grup
orig0 = string({tmpList0.orig_id}.');
[~,~,grp0] = unique(orig0);
G0 = table((1:numel(tmpList0)).', grp0, 'VariableNames',{'idx','grp'});

%% === 1 SINIFI: 1 sn standardizasyon + GRUP TESPİTİ (fast_numeric, audioinfo YOK) ===
fprintf('\n=== 1 SINIFI: 1 sn standardizasyon + GRUP TESPİTİ (%s) ===\n', groupMode)
ads1   = audioDatastore(dir1,'IncludeSubfolders',true, ...
           'FileExtensions',{'.wav'});   % sadece WAV
files1 = ads1.Files;

% Dosya adından 7 haneli numarayı çek
pat  = digitsPattern(7);
nums = nan(numel(files1),1);
for i=1:numel(files1)
    [~,b,~] = fileparts(files1{i});
    ex = extract(b, pat);
    if ~isempty(ex), nums(i) = str2double(ex(1)); end
end
% Sıralama isteğe bağlı (grup ataması numaradan)
% [nums,ord] = sort(nums); files1 = files1(ord);  % istersen aç

% Grup id (fast_numeric)
if all(isnan(nums))
    grpIdx = (1:numel(files1))';
else
    minN   = min(nums(~isnan(nums)));
    rough  = floor((nums - minN)/1000);        % yaklaşık her 1000 numara -> 1 grup
    nanmsk = isnan(rough);
    if any(~isnan(rough))
        maxr = max(rough(~nanmsk));
    else
        maxr = 0;
    end
    rough(nanmsk) = maxr + (1:sum(nanmsk));
    grpIdx = rough(:);
end

% Class-0 ve Class-1 için ayrı grup evrenlerinden split hesapla (80/10/10)
uGrp0 = unique(G0.grp);
uGrp1 = unique(grpIdx);
[tr0,va0,te0] = split_groups(uGrp0,0.8,0.1,0.1);
[tr1,va1,te1] = split_groups(uGrp1,0.8,0.1,0.1);

% Class-1 manifest (yazarken doldurulacak)
M1 = table(strings(0,1), strings(0,1), strings(0,1), NaN(0,1), zeros(0,1), ...
    'VariableNames', {'split','outfile','orig_id','num','seg_start_s'});

fprintf('[1] Yazım başlıyor (güvenli okuma, <1.0 s atlanır)...\n')
tStart1 = tic; kept = 0; skipped = 0;

badLog = fullfile(outDir,'class1_badfiles.txt');
if exist(badLog,'file'), delete(badLog); end

for i = 1:numel(files1)
    f = files1{i};
    g = grpIdx(i);

    % Split seçimi
    if ismember(g,tr1), split="train";
    elseif ismember(g,va1), split="val";
    else, split="test";
    end

    % --- OKUNABİLİRLİK KONTROLÜ (hızlı)
    if exist(f,'file') ~= 2
        skipped = skipped + 1;
        if mod(skipped,100)==0, fprintf('[1] skip(missing) %d\n', skipped); end
        appendline(badLog, sprintf('MISSING: %s', f));
        continue
    end
    fid = fopen(f,'r');
    if fid < 0
        skipped = skipped + 1;
        appendline(badLog, sprintf('FOPEN_FAIL: %s', f));
        continue
    else
        fclose(fid);
    end

    % --- GÜVENLİ OKUMA (2 deneme)
    ok = false; x = []; Fs = [];
    try
        [x,Fs] = audioread(f);
        ok = true;
    catch
        % Bazı sistemlerde string path sorun çıkarıyor → char ile tekrar dene
        try
            [x,Fs] = audioread(char(f));
            ok = true;
        catch ME
            ok = false;
            appendline(badLog, sprintf('AUDIOREAD_FAIL: %s | %s', f, ME.message));
        end
    end
    if ~ok
        skipped = skipped + 1;
        if mod(skipped,100)==0
            fprintf('[1] skip(read-fail) %d | i=%d\n', skipped, i);
        end
        continue
    end

    % Mono + resample + 1 sn kontrol
    if size(x,2)>1, x = mean(x,2); end
    if Fs~=FsT, x = resample(x,FsT,Fs); end
    if numel(x) < win
        % <1.0 s ise atla (senin kuralın)
        continue
    end

    % Normalize ve ilk 1 sn'yi yaz
    x = x ./ max(1e-9, max(abs(x)));
    seg = x(1:win);

    [~,base,~] = fileparts(f);
    nm = nums(i);  % NaN olabilir
    outName = sprintf('1_%s_%07d.wav', base, isnan(nm)*-1 + (~isnan(nm))*nm);
    outPath = fullfile(outDir, split, "1", outName);
    audiowrite(outPath, seg, FsT);

    % Manifest satırı
    M1 = [M1; {split, string(outPath), string(base), nm, 0}]; %#ok<AGROW>
    kept = kept + 1;

    if mod(i,5000)==0 || i==numel(files1)
        fprintf('[1-WRITE] %d/%d | kept:%d | skipped:%d | t=%.1fs\n', ...
            i, numel(files1), kept, skipped, toc(tStart1));
    end
end
fprintf('[1] TAMAM. kept=%d | skipped=%d | süre=%.1fs\n', kept, skipped, toc(tStart1));

% --- küçük yardımcı (dosyaya satır ekle)
function appendline(p, s)
    fid = fopen(p,'a');
    if fid>=0
        fprintf(fid, '%s\n', s);
        fclose(fid);
    end
end


%% === Class-0 WAV yazımı (split'e göre) ===
fprintf('\n=== Class 0 WAV yazımı başlıyor ===\n')
w0 = 0;
for i=1:numel(tmpList0)
    g = G0.grp(i);
    if ismember(g,tr0), split="train";
    elseif ismember(g,va0), split="val";
    else, split="test";
    end

    parts = tmpList0(i).parts;
    [~,base,~] = fileparts(parts{1}.fpath);
    for k=1:numel(parts)
        [x,Fs] = audioread(parts{k}.fpath);
        if size(x,2)>1, x = mean(x,2); end
        if Fs~=FsT, x = resample(x,FsT,Fs); end
        x = x ./ max(1e-9, max(abs(x)));
        seg = x((k-1)*win+1:k*win);

        outName = sprintf('0_%s_%04d.wav', base, k);
        outPath = fullfile(outDir, split, "0", outName);
        audiowrite(outPath, seg, FsT);

        M0 = [M0; {split, string(outPath), "0", string(base), (k-1)*segSec}]; %#ok<AGROW>
        w0 = w0 + 1;
    end
    if mod(i,100)==0 || i==numel(tmpList0)
        fprintf('[0-WRITE] %d/%d dosya | toplam seg:%d\n', i, numel(tmpList0), w0);
    end
end

%% === MANIFEST CSV'LER ===
fprintf('\n=== Manifest yazılıyor ===\n')
% Ortak şema: split, outfile, label, orig_id, seg_start_s
T0 = M0; T0.label = repmat("0",height(T0),1);
T1 = table(M1.split, M1.outfile, repmat("1",height(M1),1), M1.orig_id, M1.seg_start_s, ...
           'VariableNames', {'split','outfile','label','orig_id','seg_start_s'});
All = [T0; T1];

for s = folders
    mask  = All.split==s;
    Ts    = All(mask,:);
    outCsv = fullfile(outDir, s + "_manifest.csv");
    writetable(Ts, outCsv);
    fprintf('  -> %s (%d satır)\n', outCsv, height(Ts));
end

fprintf('\nBİTTİ ✅  Çıkış klasörü: %s\n', outDir)
end

% ===== yardımcı: grup split =====
function [tr,va,te] = split_groups(u, pTr, pVa, pTe)
u   = u(:);
n   = numel(u);
idx = randperm(n);
nTr = round(pTr*n);
nVa = round(pVa*n);
tr  = u(idx(1:nTr));
va  = u(idx(nTr+1:nTr+nVa));
te  = u(idx(nTr+nVa+1:end));
end
