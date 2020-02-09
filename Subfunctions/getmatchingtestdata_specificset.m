function [respmat, labels, training_set]= getmatchingtestdata_specificset(net, LOI, NSind, iterforeachN, ...
    image_sets_standard_tmp, image_sets_control1_tmp, image_sets_control2_tmp, ...
    image_sets_standard_tmpT, image_sets_control1_tmpT, image_sets_control2_tmpT, setNo, number_sets)

%% Get sample and test images
sample_images = [];tmp = cat(5, image_sets_standard_tmp, image_sets_control1_tmp, image_sets_control2_tmp);
for ss = 1:length(setNo)
    setind = setNo(ss); % set used in the comparison task
    sample_images = cat(3, sample_images, squeeze(tmp(:,:,:,:,setind))); 
end


test_images = [];tmp = cat(5, image_sets_standard_tmpT, image_sets_control1_tmpT, image_sets_control2_tmpT);
for ss = 1:length(setNo)
    setind = setNo(ss);
    test_images = cat(3, test_images, squeeze(tmp(:,:,:,:,setind))); 
end
% [image_sets_standard_tmp, image_sets_control1_tmp, image_sets_control2_tmp]...
%     = Stimulus_generation_Nasr(number_sets_test, image_iter);
% if setNo == 1
%     nonmatch_images = image_sets_standard_tmp;
% elseif setNo ==2
%     nonmatch_images = image_sets_control1_tmp;
% elseif setNo ==3
%     nonmatch_images = image_sets_control2_tmp;
% end


training_set = zeros(size(image_sets_standard_tmp,1), size(image_sets_standard_tmp,2), iterforeachN, length(number_sets), 2);
training_Nums = zeros(2,iterforeachN, length(number_sets));
training_labels = zeros(iterforeachN, length(number_sets));
for ii = 1:length(number_sets)
    Nsample = ii;
    tmp = 1:length(number_sets); tmp(tmp==ii) = [];
    for jj = 1:iterforeachN
    Ntest = datasample(tmp, 1);
    indrand = randi(size(sample_images, 3), [1,2]);

    imgtmp1 = squeeze(sample_images(:,:,indrand(1), Nsample));
    imgtmp2 = squeeze(test_images(:,:,indrand(2), Ntest));
    training_set(:,:,jj,ii, 1) = imgtmp1;
    training_set(:,:,jj,ii, 2) = imgtmp2;
    training_Nums(1,jj,ii) = Nsample;
    training_Nums(2,jj,ii) = Ntest;
    training_labels(jj,ii) = (training_Nums(2,jj,ii)-training_Nums(1,jj,ii))>0;
    end
end

%% get activation

response_tot_sample = getactivation(net, LOI, squeeze(training_set(:,:,:,:,1)));
response_tot_test = getactivation(net, LOI, squeeze(training_set(:,:,:,:,2)));
response_tot_NS_sample = response_tot_sample(:,:,NSind);
response_tot_NS_test = response_tot_test(:,:,NSind);
respmat = cell(1,2);
respmat{1} = response_tot_NS_sample;
respmat{2} = response_tot_NS_test;
labels = cell(1,2);
labels{1} = training_Nums;
labels{2} = training_labels;

end