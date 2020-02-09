function [X, Y] = getdataformat(resp_matching)

response_tot_NS_sample = resp_matching{1};
response_tot_NS_match = resp_matching{2};
response_tot_NS_nonmatch = resp_matching{3};
matching_inputs = cat(3, response_tot_NS_sample, response_tot_NS_match);
nonmatching_inputs = cat(3, response_tot_NS_sample, response_tot_NS_nonmatch);

% randind = randi(100);
% tmp1 = squeeze(response_tot_NS_sample(1,randind, :));
% tmp2 = squeeze(response_tot_NS_sample(30,randind, :));
% errorbar([1,30], [mean(tmp1), mean(tmp2)], [std(tmp1), std(tmp2)])

matching_inputs2D = zeros(size(matching_inputs, 3), size(matching_inputs, 1)*size(matching_inputs, 2));
nonmatching_inputs2D = zeros(size(nonmatching_inputs, 3), size(nonmatching_inputs, 1)*size(nonmatching_inputs, 2));
kk = 1;
for ii = 1:size(matching_inputs, 1)
    for jj = 1: size(matching_inputs, 2)
        matching_inputs2D(:,kk) = squeeze(matching_inputs(ii,jj, :));
        nonmatching_inputs2D(:,kk) = squeeze(nonmatching_inputs(ii,jj, :));
        kk = kk+1;
    end
end

%% define dataset 
XTraintmp = cat(2,matching_inputs2D, nonmatching_inputs2D);
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