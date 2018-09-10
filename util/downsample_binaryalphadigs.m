clear

if ~exist('binaryalphadigs.mat', 'file') == 2
    urlwrite('https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat', 'binaryalphadigs.mat')
end
load('binaryalphadigs.mat')

num_per_class = 39;
num_train = 30;
num_test = num_per_class - num_train;
num_classes = 36;

dat_small = cell(num_classes, num_per_class);
for i=1:1404
    dat_small{i} = imresize(dat{i}, 0.5);
end
dat_small = cell2table(dat_small);

train_arr = zeros(num_classes*num_train, 80);
test_arr= zeros(num_classes*num_test, 80);

for i=1:num_classes
    idx = randsample(num_per_class, num_train, false);
    choice = zeros(1, num_per_class);
    choice(idx) = 1;
    choice = logical(choice);
    
    train = dat_small{i, :}(choice);
    test = dat_small{i, :}(~choice);
    for j=1:num_train
        train_arr(j + (i-1)*num_train, :) = reshape(train{j}, 1, 80);
    end
    
    for k=1:num_test
        test_arr(k + (i-1)*num_test, :) = reshape(test{k}, 1, 80);
    end
end

csvwrite('binaryalphadigs_train.csv', train_arr)
csvwrite('binaryalphadigs_test.csv', test_arr)