function [NSind, units_PN] = getNSind(net)

isANOVA1 = 0; 
number_sets = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30];

%% Step 1. load pretrained networks
%% Sample : alexnet
layers_set = {'relu1', 'relu2', 'relu3', 'relu4', 'relu5'};

%% Step 2. generate stimulus set
image_iter = 50;
%image_iter = 50;
[image_sets_standard, image_sets_control1, image_sets_control2]...
    = Stimulus_generation_Nasr(number_sets, image_iter);

%% Step 3. define layer of interest and calculate response to stimulus
LOI = layers_set{5}; %  Layer of interest (layer 5)
response_tot_standard = getactivation(net, LOI, image_sets_standard);
response_tot_control1 = getactivation(net, LOI, image_sets_control1);
response_tot_control2 = getactivation(net, LOI, image_sets_control2);
% get total response matrix
response_tot = cat(2,response_tot_standard, response_tot_control1, response_tot_control2);
%% Step 4. get p-values from response

pvalues = getpv(response_tot); % 2 way ANOVA

%% Step 5. Analyze p-values to find number selective neurons
p_th = 0.01;p_th2 = 0.01;p_th3 = 0.01;
if isANOVA1
pv1 = pvalues(1,:); pv2 = pvalues(2,:);pv3 = pvalues(3,:); 
pv5 = pvalues2(1,:);pv6 = pvalues2(3,:);
pv4 = pvalues2(2,:); 
ind1 = (pv1<p_th);
ind2 = (pv2>p_th2);
ind3 = (pv3>p_th2);
ind4 = pv4<p_th3; % 1-way anova for area
ind5 = pv5<p_th3;ind6 = pv6<p_th3;
ind_NS = find(ind1.*ind2.*ind3.*ind4.*ind5.*ind6); % indices of number selective units
else
pv1 = pvalues(1,:); pv2 = pvalues(2,:);pv3 = pvalues(3,:); 
ind1 = (pv1<p_th);
ind2 = (pv2>p_th2);
ind3 = (pv3>p_th2);
ind_NS = find(ind1.*ind2.*ind3); % indices of number selective units
end

%% Step 6. get mean tuning curves
resp_mean = squeeze((mean(response_tot, 2))); resp_std = squeeze(std(response_tot, 0,2));
resp_mean_standard = squeeze((mean(response_tot_standard, 2))); resp_std_standard = squeeze(std(response_tot_standard, 0,2));
resp_mean_control1 = squeeze((mean(response_tot_control1, 2))); resp_std_control1 = squeeze(std(response_tot_control1, 0,2));
resp_mean_control2 = squeeze((mean(response_tot_control2, 2))); resp_std_control2 = squeeze(std(response_tot_control2, 0,2));

%% Step 7. get preferred number
response_NS_tot = response_tot(:,:,ind_NS);
response_NS_mean = squeeze(mean(response_NS_tot, 2));
[M,PNind] = max(response_NS_mean); %% PNind : preferred number of number selective neurons

units_N = length(pv1);units_PN = zeros(1,units_N)/0;
units_PN(ind_NS) = PNind;

NSind = find(units_PN>0);
end