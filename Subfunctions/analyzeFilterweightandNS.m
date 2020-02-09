function [bias_foreachfilter, inc_foreachfilter, ...
    dec_foreachfilter, var_foreachfilter, NS_foreachfilter] ...
    = analyzeFilterweightandNS(net_changed, NS_PN, layers_OI, array_sz, layers_ind_conv)


    for lind = layers_OI
        %% get weight bias for each filter
        filteroflayer = (net_changed.Layers(layers_ind_conv(lind)).Weights);
        
        bias_foreachfilter = zeros(1,size(filteroflayer, 4));
        inc_foreachfilter = zeros(1,size(filteroflayer, 4));
        dec_foreachfilter = zeros(1,size(filteroflayer, 4));
        var_foreachfilter = zeros(1,size(filteroflayer, 4));
        NS_foreachfilter = zeros(1,size(filteroflayer, 4));
        
        for ii = 1:size(filteroflayer, 4)
            filtertmp = squeeze(filteroflayer(:,:,:,ii));
            bias_foreachfilter(ii) = mean(filtertmp(:));
            var_foreachfilter(ii) = std(filtertmp(:));
        end
        
        sz = array_sz(lind, :);
        
        indtmp = find(NS_PN ==1);
        [I1, I2, I3] = ind2sub(sz, indtmp);
        for ii = 1:size(filteroflayer, 4)
            dec_foreachfilter(ii) = sum(I3==ii);
        end
        
        indtmp = find(NS_PN ==16);
        [I1, I2, I3] = ind2sub(sz, indtmp);
        for ii = 1:size(filteroflayer, 4)
            inc_foreachfilter(ii) = sum(I3==ii);
        end
        
        indtmp = find(NS_PN>0);
        [I1, I2, I3] = ind2sub(sz, indtmp);
        for ii = 1:size(filteroflayer, 4)
            NS_foreachfilter(ii) = sum(I3==ii)/(sz(1)*sz(2));
        end
    end   



end