clc;
clear;
rng(2025);   % İstersen değiştir, rastgelelik sabit kalsın

posRoot = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\İha Akustik ses\Veri_Setleri\veri_seti_dads\1";
outRoot = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V3_SON_TEST";
outPosFolder = fullfile(outRoot, "1");

if ~exist(outPosFolder, "dir")
    mkdir(outPosFolder);
end

fprintf("Pozitif hedef klasor: %s\n\n", outPosFolder);
adsPosAll = audioDatastore(posRoot, ...
    "IncludeSubfolders", true, ...
    "FileExtensions", ".wav");

allFiles = adsPosAll.Files;
nAll = numel(allFiles);
fprintf("Pozitif havuzdaki toplam dosya sayisi: %d\n\n", nAll);

firstID   = 17000;   
lastID    = 179999;   
groupWidth = 1000;    % her drone icin 1000 parca

nGroups = floor((lastID - firstID + 1) / groupWidth);
fprintf("Toplam drone grubu (1000'lik blok) sayisi: %d\n\n", nGroups);

groupFiles = cell(nGroups, 1);

for i = 1:nAll
    f = allFiles{i};
    [~, name, ~] = fileparts(f);
    
    tokens = regexp(name, '(\d+)', 'tokens');
    if isempty(tokens)
        % isimde sayi yoksa atla
        continue;
    end

    idStr = tokens{end}{1};
    idNum = str2double(idStr);

    if isnan(idNum)
        continue;
    end
    
    if idNum < firstID || idNum > lastID
        continue;   % bizim drone bloklarımızın dışında
    end

    grpIdx = floor((idNum - firstID) / groupWidth) + 1;

    if grpIdx < 1 || grpIdx > nGroups
        continue;
    end

    groupFiles{grpIdx} = [groupFiles{grpIdx}; {f}];
end

for g = 1:nGroups
    fprintf("Grup %3d icindeki dosya sayisi: %4d\n", g, numel(groupFiles{g}));
end
fprintf("\n");

targetPosTotal = 3000;

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

save(fullfile(outRoot, "Test2_pos_filelist.mat"), "testPosFiles");

for i = 1:numel(testPosFiles)
    src = testPosFiles{i};
    [~, name, ext] = fileparts(src);
    
    dst = fullfile(outPosFolder, sprintf("pos2_%04d_%s%s", i, name, ext));
    copyfile(src, dst);
end

fprintf("TUM pozitif dosyalar 1 klasorune kopyalandi.\n");
fprintf("\n== POS TEST2 OLUSTURMA BITTI ==\n");

