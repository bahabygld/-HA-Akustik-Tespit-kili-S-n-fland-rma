%% pos_test2_olustur.m
clc;
clear;
rng(2025);   % İstersen değiştir, rastgelelik sabit kalsın

%% 1) KAYNAK VE HEDEF KLASÖRLERİ TANIMLA

% Pozitif kaynak (drone var)
posRoot = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\İha Akustik ses\Veri_Setleri\veri_seti_dads\1";

% Yeni test setinin yazılacağı ana klasör
outRoot = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V3_SON_TEST";

% Pozitif sınıfın hedef klasörü (1)
outPosFolder = fullfile(outRoot, "1");

% Klasör yoksa oluştur
if ~exist(outPosFolder, "dir")
    mkdir(outPosFolder);
end

fprintf("Pozitif hedef klasor: %s\n\n", outPosFolder);

%% 2) TUM POZITIF ADAY .WAV DOSYALARINI OKU

adsPosAll = audioDatastore(posRoot, ...
    "IncludeSubfolders", true, ...
    "FileExtensions", ".wav");

allFiles = adsPosAll.Files;
nAll = numel(allFiles);
fprintf("Pozitif havuzdaki toplam dosya sayisi: %d\n\n", nAll);

%% 3) ID ARALIGINI VE GRUP YAPISINI TANIMLA

% Senin tanımına göre:
% drone 1 : 0017000–0017999
% drone 2 : 0018000–0018999
% ...
% son drone : 01790000 civarı
% Biz ID'yi sayi olarak 17000–179999 aralığında alıyoruz.

firstID   = 17000;    % 0017000
lastID    = 179999;   % 0179999
groupWidth = 1000;    % her drone icin 1000 parca

% Toplam grup sayısını hesapla
nGroups = floor((lastID - firstID + 1) / groupWidth);
fprintf("Toplam drone grubu (1000'lik blok) sayisi: %d\n\n", nGroups);

% Her grup için dosya listesi tutacağımız cell
groupFiles = cell(nGroups, 1);

%% 4) TUM DOSYALARI DOLAS, ISIMDEN NUMARA CEK, GRUBA ATA

for i = 1:nAll
    f = allFiles{i};
    [~, name, ~] = fileparts(f);

    % İsim içindeki sayıları yakala (0017000 gibi)
    tokens = regexp(name, '(\d+)', 'tokens');
    if isempty(tokens)
        % isimde sayi yoksa atla
        continue;
    end

    % Son bulunan sayıyı ID olarak al (genelde sondaki 0017000 gibi)
    idStr = tokens{end}{1};
    idNum = str2double(idStr);

    if isnan(idNum)
        continue;
    end

    % İlgili ID aralığında mı?
    if idNum < firstID || idNum > lastID
        continue;   % bizim drone bloklarımızın dışında
    end

    % Grup indexi: 0017000–0017999 -> 1, 0018000–0018999 -> 2, ...
    grpIdx = floor((idNum - firstID) / groupWidth) + 1;

    if grpIdx < 1 || grpIdx > nGroups
        continue;
    end

    groupFiles{grpIdx} = [groupFiles{grpIdx}; {f}];
end

% Grupların doluluklarını görmek için (kontrol amaçlı)
for g = 1:nGroups
    fprintf("Grup %3d icindeki dosya sayisi: %4d\n", g, numel(groupFiles{g}));
end
fprintf("\n");

%% 5) HER GRUPTAN EŞIT SAYIDA RANDOM DOSYA SEÇ (TOPLAM ~3000)

targetPosTotal = 3000;

% 3000'e en yakın bölünmeyi bul
basePerGroup = floor(targetPosTotal / nGroups);   % örn: 18
altPerGroup  = ceil(targetPosTotal / nGroups);    % örn: 19

totalLow  = basePerGroup * nGroups;
totalHigh = altPerGroup  * nGroups;

if abs(totalLow - targetPosTotal) <= abs(totalHigh - targetPosTotal)
    nPerGroup = basePerGroup;   % 18
    finalTotal = totalLow;
else
    nPerGroup = altPerGroup;    % 19
    finalTotal = totalHigh;
end

fprintf("Her gruptan %d adet secilecek. Beklenen toplam secim: %d (hedef: %d)\n\n", ...
    nPerGroup, finalTotal, targetPosTotal);

% Secilen dosyalar
selectedPosFiles = {};

for g = 1:nGroups
    filesG = groupFiles{g};
    nG = numel(filesG);

    if nG == 0
        warning("Grup %d bos, atlaniyor.", g);
        continue;
    end

    if nG < nPerGroup
        warning("Grup %d icin yeterli dosya yok (%d adet), hepsini aliyorum.", g, nG);
        nTake = nG;
    else
        nTake = nPerGroup;
    end

    idxG = randperm(nG, nTake);
    selectedPosFiles = [selectedPosFiles; filesG(idxG)];
end

fprintf("Gercekten secilen pozitif dosya sayisi: %d\n\n", numel(selectedPosFiles));

testPosFiles = selectedPosFiles;

%% 6) LİSTEYİ .MAT OLARAK KAYDET

save(fullfile(outRoot, "Test2_pos_filelist.mat"), "testPosFiles");

%% 7) SEÇİLENLERİ 1 KLASÖRÜNE KOPYALA

for i = 1:numel(testPosFiles)
    src = testPosFiles{i};
    [~, name, ext] = fileparts(src);
    
    % yeni isim: pos2_0001_orjinalIsim.wav
    dst = fullfile(outPosFolder, sprintf("pos2_%04d_%s%s", i, name, ext));
    copyfile(src, dst);
end

fprintf("TUM pozitif dosyalar 1 klasorune kopyalandi.\n");
fprintf("\n== POS TEST2 OLUSTURMA BITTI ==\n");
