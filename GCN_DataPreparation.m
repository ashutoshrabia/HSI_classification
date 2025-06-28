I = double(imread('19920612_AVIRIS_IndianPine_Site3.tif'));
I = I(:,:,[1:103, 109:149,164:219]);
[m, n, z] = size(I);

TR = double(imread('IndianTR123_temp123.tif'));
TE = double(imread('IndianTE123_temp123.tif'));

I2d = hyperConvert2d(I);
for i = 1 : z
    row = I2d(i, :);
    mn  = min(row);
    mx  = max(row);
    if mx > mn
        I2d(i, :) = (row - mn) ./ (mx - mn);
    else
        I2d(i, :) = zeros(size(row));  % or leave as-is
    end
end

TR2d = hyperConvert2d(TR);
TE2d = hyperConvert2d(TE);

TR_sample = I2d(:,TR2d>0);
TE_sample = I2d(:,TE2d>0);

TR_temp = TR2d(:,TR2d>0);
TE_temp = TE2d(:,TE2d>0);

X = [TR_sample,TE_sample];
Y = [TR_temp, TE_temp];

K = 10;
si = 1;

ALL_W = creatLap(X,K, si);
ALL_D = (sum(ALL_W, 2)).^(-1/2);
ALL_D = diag(ALL_D);
L_temp = ALL_W * ALL_D;
ALL_L = ALL_D * L_temp;
ALL_L = ALL_L + eye(size(ALL_L));
ALL_L = sparse(ALL_L);

ALL_X = X';
ALL_Y = Y';

%% Please replace the following route with your own one
save('C:\Users\ashut\vyomchara\GCN/ALL_X.mat','ALL_X');
save('C:\Users\ashut\vyomchara\GCN/ALL_Y.mat','ALL_Y');
save('C:\Users\ashut\vyomchara\GCN/ALL_L.mat','ALL_L');