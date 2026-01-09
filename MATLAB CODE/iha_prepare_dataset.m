function iha_prepare_dataset()
% === KULLANICI AYARLARI ===
rootDir = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\İha Akustik ses\Veri_Setleri\veri_seti_dads";
dir0 = fullfile(rootDir,"0");
dir1 = fullfile(rootDir,"1");
FsT  = 16000;
segSec = 1.0; win = round(segSec*FsT);
minPosDur = 1.0;   % 1 sınıfında <1.0 s olanları AT
simThresh  = 0.985; % kaynak grup tespiti benzerlik eşiği
rng(42);

% === YARDIMCI ===
asWav = {'.wav','.flac','.mp3','.m4a','.ogg'};
mkds = @(p) audioDatastore(p,'IncludeSubfolders',true,'FileExtensions',asWav);

% --- 0 SINIFI: 1 sn'lik böl, son artıkları at ---
ads0 = mkds(dir0);
T0 = table(); X0 = {}; % segment dalgaları
for i=1:numel(ads0.Files)
    f = ads0.Files{i};
    try
        [x,Fs] = audioread(f);
    catch
        warning('Read fail (0): %s', f); continue;
    end
    if size(x,2)>1, x = mean(x,2); end
    if Fs~=FsT, x = resample(x,FsT,Fs); end
    L = numel(x);
    nSeg = floor(L/win);
    if nSeg<1, continue; end
    x = x(1:nSeg*win);
    x = x ./ max(1e-9, max(abs(x)));
    for k=1:nSeg
        seg = x((k-1)*win+1:k*win);
        X0{end+1,1} = seg; %#ok<AGROW>
    end
    [~,base,~] = fileparts(f);
    T0 = [T0; table(repmat(string(f),nSeg,1), repmat(string(base),nSeg,1), ...
        (0:nSeg-1)'*segSec, repmat("0",nSeg,1), ...
        'VariableNames',{'file','orig_id','seg_start_s','label'})]; %#ok<AGROW>
end

% --- 1 SINIFI: <1.0 s at; >1.0 s ise ilk 1.0 s'i al ---
ads1 = mkds(dir1);
T1_raw = table(); X1_raw = {}; num1 = [];
pat = digitsPattern(7);

files1 = ads1.Files;
% Dosya adındaki 7 haneli numarayı çıkar ve numaraya göre sırala
nums = zeros(numel(files1),1);
for i=1:numel(files1)
    [~,b,~] = fileparts(files1{i});
    ex = extract(b, pat);
    if ~isempty(ex), nums(i) = str2double(ex(1)); else, nums(i) = NaN; end
end
[~,ord] = sort(nums);
files1 = files1(ord); nums = nums(ord);

for i=1:numel(files1)
    f = files1{i};
    try
        info = audioinfo(f);
    catch
        warning('Info fail (1): %s', f); continue;
    end
    if info.Duration < minPosDur - 1e-3
        continue; % 1 s altını at
    end
    [x,Fs] = audioread(f);
    if size(x,2)>1, x = mean(x,2); end
    if Fs~=FsT, x = resample(x,FsT,Fs); end
    x = x ./ max(1e-9, max(abs(x)));
    if numel(x) < win
        continue; % güvenlik
    end
    seg = x(1:win); % 1.0 s
    X1_raw{end+1,1} = seg; %#ok<AGROW>
    [~,base,~] = fileparts(f);
    thisNum = nums(i);
    T1_raw = [T1_raw; table(string(f), string(base), 0, "1", thisNum, ...
        'VariableNames',{'file','orig_id','seg_start_s','label','num'})]; %#ok<AGROW>
    num1(end+1,1) = thisNum; %#ok<AGROW>
end

% --- 1 SINIFI: kaynak-grup (oto) — ardışık dosyalar arası benzerlik ---
% Özellik: 64 log-mel ortalamasının vektörü
fe_logmel = @(sig) mean(log(1e-6 + melSpectrogram(sig,FsT, ...
    'WindowLength',round(0.025*FsT),'OverlapLength',round(0.015*FsT), ...
    'NumBands',64,'FFTLength',1024)),2);

fprintf('Computing log-mel means for class 1...\n');
N1 = numel(X1_raw);
LM = zeros(64,N1,'single');
for i=1:N1, LM(:,i) = fe_logmel(X1_raw{i}); end

% Kosinüs benzerliği ardışık örnekler arasında
cosSim = @(a,b) (a'*b) / (max(1e-9,norm(a))*max(1e-9,norm(b)));
isBreak = true(N1,1); % ilk kayıt yeni grup
for i=2:N1
    s = cosSim(LM(:,i-1), LM(:,i));
    isBreak(i) = s < simThresh;
end
grp = cumsum(isBreak);  % grup id
T1 = T1_raw;
T1.grp = grp;

% --- 0 SINIFI için grup tanımı ---
% Her kaynak dosya başlı başına grup (leakage olmasın)
T0.grp = grp_from_orig(T0.orig_id);

% --- BİRLEŞTİR ---
X = [X0; X1_raw];
T = [T0; removevars(T1,{'num'})];

% --- 158D ÖZELLİK ÇIKAR ---
fprintf('Extracting 158D features...\n');
N = numel(X);
F = zeros(N, 64+64 + 13+13 + 3, 'single'); % 158
for i=1:N
    F(i,:) = feat158(X{i}, FsT);
    if mod(i,10000)==0, fprintf('  %d/%d\n', i, N); end
end
Tfeat = array2table(F);
Tfeat.file = T.file;
Tfeat.orig_id = T.orig_id;
Tfeat.seg_start_s = T.seg_start_s;
Tfeat.label = categorical(T.label);
Tfeat.grp = T.grp;

% --- GROUP-AWARE SPLIT (80/10/10), stratified by label majority ---
G = groupsummary(Tfeat, 'grp', 'mode', 'label');
% stratify by mode(label)
uLabs = categories(Tfeat.label);
% split her sınıftaki grupları ayrı ayrı
tr = false(height(G),1); va = false(height(G),1); te = false(height(G),1);
for c = 1:numel(uLabs)
    lab = uLabs{c};
    idx = G.mode_label == lab;
    ids = find(idx);
    ids = ids(randperm(numel(ids)));
    n = numel(ids);
    nTr = round(0.8*n); nVa = round(0.1*n);
    tr(ids(1:nTr)) = true;
    va(ids(nTr+1:nTr+nVa)) = true;
    te(ids(nTr+nVa+1:end)) = true;
end

tr_grp = G.grp(tr); va_grp = G.grp(va); te_grp = G.grp(te);
isTr = ismember(Tfeat.grp, tr_grp);
isVa = ismember(Tfeat.grp, va_grp);
isTe = ismember(Tfeat.grp, te_grp);

Train = Tfeat(isTr,:); Val = Tfeat(isVa,:); Test = Tfeat(isTe,:);

% --- EĞİTİM SETİNDE 1:1 DENGELEME (undersample) ---
pos = Train(Train.label=="1",:);
neg = Train(Train.label=="0",:);
m = min(height(pos), height(neg));
idxP = randperm(height(pos), m);
idxN = randperm(height(neg), m);
Train_bal = [pos(idxP,:); neg(idxN,:)];
Train_bal = Train_bal(randperm(height(Train_bal)),:);

% --- KAYITLAR ---
writetable(strip_for_csv(Train_bal), "train.csv");
writetable(strip_for_csv(Val),       "val.csv");
writetable(strip_for_csv(Test),      "test.csv");

% Grup raporu (kontrol amaçlı)
GR0 = groupsummary(Tfeat(Tfeat.label=="0",:), 'grp', 'count');
GR1 = groupsummary(Tfeat(Tfeat.label=="1",:), 'grp', 'count');
GR0.label = repmat("0",height(GR0),1);
GR1.label = repmat("1",height(GR1),1);
GR = [GR0; GR1];
writetable(GR, "group_report.csv");

fprintf('\nDONE.\n');
fprintf('Train_bal: %d rows | Val: %d | Test: %d\n', height(Train_bal), height(Val), height(Test));

end % main

% ===== Helpers =====

function gid = grp_from_orig(orig_id)
% Aynı orig_id (kaynak dosya) aynı grupta — leak engeller
[u,~,ic] = unique(orig_id);
gid = ic;
end

function feats = feat158(x,Fs)
x = x(:);
f = melSpectrogram(x,Fs, ...
    'WindowLength',round(0.025*Fs), 'OverlapLength',round(0.015*Fs), ...
    'NumBands',64,'FFTLength',1024);
f = log(1e-6 + f);             
m_mean = mean(f,2); m_std = std(f,0,2);
coeffs = mfcc(x,Fs,'LogEnergy','Ignore'); % [T x 13]
c_mean = mean(coeffs,1).'; c_std = std(coeffs,0,1).';
z = mean(zerocrossrate(x));     % frame ortalaması
r = rms(x);
C = spectralCentroid(x,Fs); sc = mean(C);
feats = single([m_mean; m_std; c_mean; c_std; z; r; sc]);
end

function T = strip_for_csv(Tin)
% Özellik kolonlarını F1..F158 olarak adlandır
X = Tin{:,1:158};
T = array2table(X);
T.Properties.VariableNames = compose('F%d',1:158);
T.file = Tin.file;
T.orig_id = Tin.orig_id;
T.seg_start_s = Tin.seg_start_s;
T.label = Tin.label;
T.grp = Tin.grp;
end
