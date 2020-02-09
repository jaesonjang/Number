function [response_NS_sep, units_PN, net_perm] = ...
    generatePretrainedPermuteddataonce(LOI, number_sets, ...
    rand_layers_ind, image_iter, net, p_th1, p_th2, p_th3)

net_perm = Randomizeweight_permute(net, rand_layers_ind);
nets = {net, net_perm};
for netind = 1:length(nets)
    nettmp = nets{netind};
    [image_sets_standard, image_sets_control1, image_sets_control2]...
        = Stimulus_generation_Nasr(number_sets, image_iter);
    response_tot_standard = getactivation(nettmp, LOI, image_sets_standard);
    response_tot_control1 = getactivation(nettmp, LOI, image_sets_control1);
    response_tot_control2 = getactivation(nettmp, LOI, image_sets_control2);
    response_tot = cat(2,response_tot_standard, response_tot_control1, response_tot_control2);
    pvalues = getpv(response_tot); % 2 way ANOVA
    pv1 = pvalues(1,:); pv2 = pvalues(2,:);pv3 = pvalues(3,:);
    
    ind1 = (pv1<p_th1);
    ind2 = (pv2>p_th2);
    ind3 = (pv3>p_th3);
    ind_NS = find(ind1.*ind2.*ind3); % indices of number selective units
    
    response_NS_tot = response_tot(:,:,ind_NS);
    response_NS_mean = squeeze(mean(response_NS_tot, 2));
    [M,PNind] = max(response_NS_mean); %% PNind : preferred number of number selective neurons
    tmp1 = response_NS_tot(:,1:50, :); tmp1 = (mean(tmp1, 2));
    tmp2 = response_NS_tot(:,51:100, :);tmp2 = (mean(tmp2, 2));
    tmp3 = response_NS_tot(:,101:150,:);tmp3 = (mean(tmp3, 2));
    response_NS_sep = cat(2,tmp1, tmp2, tmp3);
    units_N = length(pv1);
    units_PN = zeros(1,units_N)/0; units_PN(ind_NS) = PNind; % preferred number for each neuron
    
end


end