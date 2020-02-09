function [blkratio, baseblkratio] = getblkratio(ind_layer, number_sets)

layers_set = {'relu1', 'relu2', 'relu3', 'relu4', 'relu5'};
iter = 100;
%ind_layer = 4;
blkratio = zeros(iter, length(number_sets));
baseblkratio = zeros(1,iter);
LOI = layers_set{ind_layer};
    
for iterind = 1:iter
    
    tic
    load(['D:\MATLAB\Number selectivity\191203_data_backtracking\data191123_iter_' num2str(iterind)])

    blank = zeros(227, 227,1,1);
    response_tot_blank = squeeze(getactivation(net_changed, LOI, blank));
    for ii = 1:length(number_sets)
        indtmp = NS_PN_L4 ==ii;
        blkrsptmp = response_tot_blank(indtmp);
        blkratio(iterind, ii) = sum(blkrsptmp>0)/length(blkrsptmp);
        baseblkratio(iterind) = sum(response_tot_blank>0)/length(response_tot_blank);
    end
    toc
end
end