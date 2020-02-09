function generateComparisontaskdata2(iter, net, rand_layers_ind, ...
    iterforeachN, iterforeachN_val,  setNo, LOI, savedir, p_th1, p_th2, p_th3, ...
    image_sets_standard, image_sets_control1,image_sets_control2, ...
    image_sets_standardT, image_sets_control1T,image_sets_control2T, number_sets)


[NSind_pretrained, units_PN_pretrained] = getNSind2(net, image_sets_standard, image_sets_control1,image_sets_control2...
    , p_th1, p_th2, p_th3, LOI); %% Calculating indices of number selective neurons in pretrained network ~ a few min
    
for iterind = 1:iter
    tic
    net_rand_permtmp = Randomizeweight_permute(net, rand_layers_ind); %% randomized network
    [NSind_permuted, units_PN_permuted] = getNSind2(net_rand_permtmp, image_sets_standard, image_sets_control1, image_sets_control2...
        , p_th1, p_th2, p_th3, LOI); %% Calculating indices number selective neurons in permuted network
    
    NNSind_pretrained = find(~(units_PN_pretrained>0));
    NNSind_permuted = find(~(units_PN_permuted>0));
    indtot = 1:length(units_PN_pretrained);
    
    [resp_pretrained, label_test, imgset_pret] = getmatchingtestdata_specificset(net, LOI, indtot, iterforeachN, ...
        image_sets_standard, image_sets_control1, image_sets_control2, ...
        image_sets_standardT, image_sets_control1T, image_sets_control2T, setNo, number_sets); %% generate training responses ~ 5 min
    
    [resp_pretrained_val, label_val, imgset_pretval] = getmatchingtestdata_specificset(net, LOI, indtot, iterforeachN_val, ...
        image_sets_standard, image_sets_control1, image_sets_control2, ...
        image_sets_standardT, image_sets_control1T, image_sets_control2T, setNo, number_sets); %% generate validation responses
    
    [resp_permuted, label_test_perm, imgset_perm] = getmatchingtestdata_specificset(net_rand_permtmp, LOI, indtot, iterforeachN, ...
        image_sets_standard, image_sets_control1, image_sets_control2, ...
        image_sets_standardT, image_sets_control1T, image_sets_control2T, setNo, number_sets); %% generate training responses ~ 5 min
    
    [resp_permuted_val, label_val_perm, imgset_permval] = getmatchingtestdata_specificset(net_rand_permtmp, LOI, indtot, iterforeachN_val, ...
        image_sets_standard, image_sets_control1, image_sets_control2, ...
        image_sets_standardT, image_sets_control1T, image_sets_control2T, setNo, number_sets); %% generate validation responses
    
    save([savedir '\Data_for_Comparisontask_onlyset2_iter_' num2str(iterind)], ...
        'NSind_pretrained', 'units_PN_pretrained', 'NSind_permuted', 'units_PN_permuted', ...
        'resp_pretrained', 'label_test', 'resp_pretrained_val', 'label_val', ...
        'resp_permuted', 'label_test_perm', 'resp_permuted_val', 'label_val_perm','imgset_pret', 'imgset_pretval', 'imgset_perm', 'imgset_permval', '-v7.3')
    toc
    disp(['[' num2str(iterind) '/' num2str(iter) ']'])
end


end