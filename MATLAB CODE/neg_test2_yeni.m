%% neg_test2_olustur.m
clc;
clear;
rng(2025);   % İstersen değiştir, rastgelelik sabit kalsın

%% 1) KAYNAK VE HEDEF KLASÖRLERİ TANIMLA

% Negatif kaynak (drone yok)
negRoot = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\İha Akustik ses\Veri_Setleri\dads_yeni\0";

% Yeni test setinin yazılacağı ana klasör
outRoot = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V3_SON_TEST";

% Negatif sınıfın hedef klasörü (0)
outNegFolder = fullfile(outRoot, "0");

% Klasör yoksa oluştur
if ~exist(outNegFolder, "dir")
    mkdir(outNegFolder);
end

fprintf("Negatif hedef klasor: %s\n\n", outNegFolder);

%% 2) TUM NEGATIF .WAV DOSYALARINI OKU

adsNeg = audioDatastore(negRoot, ...
    "IncludeSubfolders", true, ...
    "FileExtensions", ".wav");

nNeg = numel(adsNeg.Files);
fprintf("Negatif havuzdaki toplam dosya sayisi: %d\n", nNeg);

%% 3) RANDOM 3000 ADET DOSYA SEÇ

targetNeg = 2934;

if nNeg < targetNeg
    warning("Negatif dosya sayisi %d, 3000'den az. Hepsini aliyorum.", nNeg);
    targetNeg = nNeg;
end

idxNeg = randperm(nNeg, targetNeg);        % random indexler
testNegFiles = adsNeg.Files(idxNeg);       % seçilen dosyaların tam yolları

fprintf("Secilen negatif dosya sayisi: %d\n\n", numel(testNegFiles));

%% 4) SEÇİLENLERİ 0 KLASÖRÜNE KOPYALA

for i = 1:numel(testNegFiles)
    src = testNegFiles{i};
    [~, name, ext] = fileparts(src);
    
    % yeni isim: neg2_0001_orjinalIsim.wav
    dst = fullfile(outNegFolder, sprintf("neg2_%04d_%s%s", i, name, ext));
    copyfile(src, dst);
end

fprintf("TUM negatif dosyalar 0 klasorune kopyalandi.\n");

%% 5) LİSTEYİ .MAT OLARAK KAYDET (ISTERSEN SONRA KULLANIRSIN)

save(fullfile(outRoot, "Test2_neg_filelist.mat"), "testNegFiles");

fprintf("\n== NEG TEST2 OLUSTURMA BITTI ==\n");
