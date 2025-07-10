function M2 = hyperConvert2d(M3)
    % HYPERCONVERT2D  Convert an HSI cube to a (bands × pixels) matrix.
    %
    %   M2 = hyperConvert2d(M3)
    %
    %   - If M3 is m×n×p, M2 is p×(m*n).
    %   - If M3 is m×n, M2 is 1×(m*n).

    nd = ndims(M3);
    if nd<2 || nd>3
        error('Input must be m×n or m×n×p.');
    end

    [m, n, p] = deal(size(M3,1), size(M3,2), ...
                      (nd==3)*size(M3,3) + (nd==2)*1);

    % Flatten:
    if nd==3
        % permute so bands come first, then linearize the spatial dims
        M2 = reshape(permute(M3, [3 1 2]), p, m*n);
    else
        % single band: just flatten to 1 × (m*n)
        M2 = reshape(M3, 1, []);
    end
end
