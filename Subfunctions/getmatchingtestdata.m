function respmat = getmatchingtestdata(net, LOI, NSind, iterforeachN)
%% Step 2. Generate stimulus
number_sets = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30];
number_sets_test = 1:48;

% iterforeachN = 100;
image_iter = 50;
multiples = [0.4 0.7 1.3 1.6];
[image_sets_standard_tmp, image_sets_control1_tmp, image_sets_control2_tmp]...
    = Stimulus_generation_Nasr(number_sets, image_iter);
sample_images = cat(3,image_sets_standard_tmp,image_sets_control1_tmp, image_sets_control2_tmp);

[image_sets_standard_tmp, image_sets_control1_tmp, image_sets_control2_tmp]...
    = Stimulus_generation_Nasr(number_sets, image_iter);
match_images = cat(3,image_sets_standard_tmp,image_sets_control1_tmp, image_sets_control2_tmp);

[image_sets_standard_tmp, image_sets_control1_tmp, image_sets_control2_tmp]...
    = Stimulus_generation_Nasr(number_sets_test, image_iter);
nonmatch_images = cat(3,image_sets_standard_tmp,image_sets_control1_tmp, image_sets_control2_tmp);


training_set = zeros(length(number_sets), iterforeachN, size(image_sets_standard_tmp,1), size(image_sets_standard_tmp,2), 3);
for ii = 1:length(number_sets)
    Ns = number_sets(ii);
    for jj = 1:iterforeachN
    indrand = randi(size(sample_images, 3));
    imgtmp1 = squeeze(sample_images(:,:,indrand, ii));
    indrand = randi(size(match_images, 3));
    imgtmp2 = squeeze(match_images(:,:,indrand, ii));

    multmp = datasample(multiples, 1);
    Nnm = round(Ns*multmp);
    while Nnm == Ns || Nnm ==0
        multmp = datasample(multiples, 1);
        Nnm = round(Ns*multmp);
    end
    indrand = randi(size(nonmatch_images, 3));
    imgtmp3 = squeeze(nonmatch_images(:,:,indrand, Nnm));
    sample_match_nonmatch = cat(3, imgtmp1, imgtmp2, imgtmp3);
    training_set(ii,jj, :,:,:) = sample_match_nonmatch;  
    end
    disp(ii)
end

training_sets = permute(training_set, [3, 4, 2, 1, 5]);

% figure
% tmpind = randi(100); 
% for ii = 1:3
% subplot(1,3,ii);imagesc(squeeze(training_sets(:,:,tmpind, 5, ii)));axis image;colormap(gray)
% end
%% get activation

response_tot_sample = getactivation(net, LOI, squeeze(training_sets(:,:,:,:,1)));
response_tot_match = getactivation(net, LOI, squeeze(training_sets(:,:,:,:,2)));
response_tot_nonmatch = getactivation(net, LOI, squeeze(training_sets(:,:,:,:,3)));

response_tot_NS_sample = response_tot_sample(:,:,NSind);
response_tot_NS_match = response_tot_match(:,:,NSind);
response_tot_NS_nonmatch = response_tot_nonmatch(:,:,NSind); 
respmat = cell(1,3);
respmat{1} = response_tot_NS_sample;
respmat{2} = response_tot_NS_match;
respmat{3} = response_tot_NS_nonmatch;

end