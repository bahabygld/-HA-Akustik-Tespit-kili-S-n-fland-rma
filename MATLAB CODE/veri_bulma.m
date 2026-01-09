root = "C:\Users\baha_\OneDrive\Masaüstü\Yüksek Lisans\1.Sınıf 2.Dönem\İha Akustik ses\SON DENGELENMİŞ\BALANCED_V3_NOISE"; 

adsTrainPos = audioDatastore(fullfile(root,"train","pos"));
adsTrainNeg = audioDatastore(fullfile(root,"train","neg"));

adsValPos   = audioDatastore(fullfile(root,"val","pos"));
adsValNeg   = audioDatastore(fullfile(root,"val","neg"));

fprintf("TRAIN POS : %d\n", numel(adsTrainPos.Files));
fprintf("TRAIN NEG : %d\n", numel(adsTrainNeg.Files));
fprintf("VAL POS   : %d\n", numel(adsValPos.Files));
fprintf("VAL NEG   : %d\n", numel(adsValNeg.Files));
