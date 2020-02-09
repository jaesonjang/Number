
function [resp_mean_RP, units_PN_RP] = getresponsecurveforL4

number_sets = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30];
layers_set = {'relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6'};

net = alexnet;
image_iter = 50;
%image_iter = 50;
[image_sets_standard, image_sets_control1, image_sets_control2]...
    = Stimulus_generation_Nasr(number_sets, image_iter);
rand_layers_ind = [2, 6, 10, 12 14];
% rand_layers_ind = [12 14];
net_rand_perm = Randomizeweight_permute(net, rand_layers_ind);

LOI = layers_set{4}; %  interested in the final layer
response_tot_standard_RP = getactivation(net_rand_perm, LOI, image_sets_standard);
response_tot_control1_RP = getactivation(net_rand_perm, LOI, image_sets_control1);
response_tot_control2_RP = getactivation(net_rand_perm, LOI, image_sets_control2);
% get total response matrix
response_tot_RP = cat(2,response_tot_standard_RP, response_tot_control1_RP, response_tot_control2_RP);

%% Step 4. get p-values from response
pvalues_RP = getpv(response_tot_RP);
p_th = 0.01;p_th2 = 0.01;p_th3 = 0.01;
pv1 = pvalues_RP(1,:); pv2 = pvalues_RP(2,:);pv3 = pvalues_RP(3,:); 
ind1 = (pv1<p_th);
ind2 = (pv2>p_th2);
ind3 = (pv3>p_th2);
ind_NS = find(ind1.*ind2.*ind3);

response_NS_tot_RP = response_tot_RP(:,:,ind_NS);
response_NS_mean_RP = squeeze(mean(response_NS_tot_RP, 2));
[M,PNind_RP] = max(response_NS_mean_RP);

resp_mean_RP = squeeze((mean(response_tot_RP, 2))); resp_std_RP = squeeze(std(response_tot_RP, 0,2));

units_N_RP = length(pv1);
units_PN_RP = zeros(1,units_N_RP)/0; units_PN_RP(ind_NS) = PNind_RP;
end