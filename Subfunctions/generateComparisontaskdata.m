
function generateComparisontaskdata(iter, pathtmp, generatecomparisondata, net, rand_layers_ind, ...
    iterforeachN, iterforeachN_val, image_iter, setNo, LOI, savedir, p_th1, p_th2, p_th3, ...
    image_sets_standard, image_sets_control1,image_sets_control2, number_sets)

if generatecomparisondata
    [NSind_pretrained, units_PN_pretrained] = getNSind2(net, image_sets_standard, image_sets_control1,image_sets_control2...
    , p_th1, p_th2, p_th3, LOI); %% get index of number selective neurons in pretrained network ~ a few min
    for iterind = 1:iter        
        net_rand = Randomizeweight_permute(net, rand_layers_ind); %% randomizing network
        [NSind_permuted, units_PN_permuted] = getNSind2(net_rand, image_sets_standard, image_sets_control1,image_sets_control2...
        , p_th1, p_th2, p_th3, LOI); %% getting index of number selective neurons in permuted network
    
        [resp_pretrained label_test] = getmatchingtestdata2(net, LOI, NSind_pretrained, iterforeachN, image_iter, setNo, number_sets); %% generating training responses 
        disp(['              [Sub-step 1/4] Training responses of the pre-trained AlexNet were generated...'])        
        [resp_pretrained_val label_val] = getmatchingtestdata2(net, LOI, NSind_pretrained, iterforeachN_val, image_iter, setNo, number_sets); %% generating validation responses
        disp(['              [Sub-step 2/4] Validating responses of the pre-trained AlexNet were generated...'])        
        [resp_permuted, label_test_perm] = getmatchingtestdata2(net_rand, LOI, NSind_permuted, iterforeachN, image_iter, setNo, number_sets); %% generating training responses 
        disp(['              [Sub-step 3/4] Training responses of the permuted AlexNet were generated...'])    
        [resp_permuted_val, label_val_perm] = getmatchingtestdata2(net_rand, LOI, NSind_permuted, iterforeachN_val, image_iter, setNo, number_sets); %% generating validation responses
        disp(['              [Sub-step 4/4] Validating responses of the permuted AlexNet were generated...'])    
            
        save([savedir '\Data_for_Comparisontask_iter_' num2str(iterind)], ...
            'NSind_pretrained', 'units_PN_pretrained', 'NSind_permuted', 'units_PN_permuted', ...
            'resp_pretrained', 'label_test', 'resp_pretrained_val', 'label_val', ...
            'resp_permuted', 'label_test_perm', 'resp_permuted_val', 'label_val_perm','-v7.3')        
    end
end
end