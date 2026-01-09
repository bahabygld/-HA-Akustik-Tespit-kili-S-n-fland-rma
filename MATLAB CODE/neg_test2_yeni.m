
clc;
clear;
rng(2025);   % İstersen değiştir, rastgelelik sabit kalsın

negRoot = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\İha Akustik ses\Veri_Setleri\dads_yeni\0";

outRoot = "C:\Users\baha_\OneDrive\Masaüstü\BALANCED_V3_SON_TEST";

outNegFolder = fullfile(outRoot, "0");

if ~exist(outNegFolder, "dir")
    mkdir(outNegFolder);
end

fprintf("Negatif hedef klasor: %s\n\n", outNegFolder);

adsNeg = audioDatastore(negRoot, ...
    "IncludeSubfolders", true, ...
    "FileExtensions", ".wav");

nNeg = numel(adsNeg.Files);
fprintf("Negatif havuzdaki toplam dosya sayisi: %d\n", nNeg);

targetNeg = 2934;

if nNeg < targetNeg
    warning("Negatif dosya sayisi %d, 3000'den az. Hepsini aliyorum.", nNeg);
    targetNeg = nNeg;
end

idxNeg = randperm(nNeg, targetNeg);        % random indexler
testNegFiles = adsNeg.Files(idxNeg);       % seçilen dosyaların tam yolları

fprintf("Secilen negatif dosya sayisi: %d\n\n", numel(testNegFiles));

for i = 1:numel(testNegFiles)
    src = testNegFiles{i};
    [~, name, ext] = fileparts(src);

    dst = fullfile(outNegFolder, sprintf("neg2_%04d_%s%s", i, name, ext));
    copyfile(src, dst);
end

fprintf("TUM negatif dosyalar 0 klasorune kopyalandi.\n");

save(fullfile(outRoot, "Test2_neg_filelist.mat"), "testNegFiles");

fprintf("\n== NEG TEST2 OLUSTURMA BITTI ==\n");

