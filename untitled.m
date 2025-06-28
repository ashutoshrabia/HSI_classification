s = load('X_train.mat');
disp(s);
whos('-file', 'X_train.mat');
fn = fieldnames(s);
disp(fn);