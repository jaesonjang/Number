function [X, Y] = getdataformat(resp_matching)

response_tot_NS_sample = resp_matching{1};
response_tot_NS_test = resp_matching{2};

inputs = cat(3, response_tot_NS_sample, response_tot_NS_test);

% randind = randi(100);
% tmp1 = squeeze(response_tot_NS_sample(1,randind, :));
% tmp2 = squeeze(response_tot_NS_sample(30,randind, :));
% errorbar([1,30], [mean(tmp1), mean(tmp2)], [std(tmp1), std(tmp2)])

inputs2D = zeros(size(inputs, 3), size(inputs, 1)*size(inputs, 2));

kk = 1;
for ii = 1:size(inputs, 1)
    for jj = 1: size(inputs, 2)
        inputs2D(:,kk) = squeeze(inputs(ii,jj, :));
        kk = kk+1;
    end
end

%% define dataset 
XTraintmp = inputs2D;
X = zeros(size(XTraintmp, 1),1,1,size(XTraintmp, 2));
for ii = 1:size(XTraintmp, 2)
    tmp = squeeze(XTraintmp(:,ii));
    X(:,:,:, ii) = tmp;
end

Y = cell(size(X, 4), 1);
for ii = 1:size(X, 4)/2
Y{ii} = 'match';
Y{ii+size(X, 4)/2} = 'nonmatch';
end
Y = categorical(Y);


end