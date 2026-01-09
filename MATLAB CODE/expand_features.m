function Texp = expand_features(Tin, varName)
% Tin.(varName): N×1 cell (her hücre 1×D) veya N×D double olabilir.
% Çıktı: varName kaldırılır, yerine D adet sayısal sütun eklenir.

    X = Tin.(varName);

    if iscell(X)
        emptyMask = cellfun(@isempty, X);
        firstIdx = find(~emptyMask, 1, 'first');
        if isempty(firstIdx)
            error('Vektör kolonu tamamen boş: %s', varName);
        end
        D = numel(X{firstIdx});
        if any(emptyMask), X(emptyMask) = {nan(1, D)}; end
        assert(all(cellfun(@(z) numel(z)==D, X)), 'Öznitelik uzunlukları tutarsız: %s', varName);
        Xmat = cell2mat(X);          % N×D

    elseif isnumeric(X)
        if isvector(X), X = reshape(X, 1, []); end
        Xmat = X;                     % N×D
        D = size(Xmat, 2);

    else
        error('Desteklenmeyen tür: %s (%s)', varName, class(X));
    end

    featNames = strcat(varName, "_", string(1:D));
    Texp = Tin(:, setdiff(Tin.Properties.VariableNames, {varName}));
    Texp = [Texp array2table(Xmat, 'VariableNames', featNames)];
end
