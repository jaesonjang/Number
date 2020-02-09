% =========================================================================================================================================================
% Demo codes for
% "Spontaneous generation of number sense in untrained deep neural networks"
% Gwangsu Kim, Jaeson Jang, Seungdae Baek, Min Song, and Se-Bum Paik*
% 
% *Contact: sbpaik@kaist.ac.kr
%
% Prerequirements
% 1) MATLAB 2018b or later version
% 2) Installation of the Deep Learning Toolbox (https://www.mathworks.com/products/deep-learning.html)
% 3) Installation of the pretrained AlexNet (https://de.mathworks.com/matlabcentral/fileexchange/59133-deep-learning-toolbox-model-for-alexnet-network)

% This code performs a demo simulation for Fig. 4 in the manuscript.
% =========================================================================================================================================================

%% Setting options for simulation
if simulation_option == 1
    generatedat = 1; savedat = 1;
    iter = 1; % Number of networks for analysis
elseif simulation_option == 2
    % This option will reproduce the figures in the manuscript.
    % Data files that are needed to perform this option are available from the corresponding author upon reasonable request.
    
    generatedat = 0; savedat = 0;
    iter = 100;
end

%% Setting file dir
pathtmp = pwd; % please set dir of Codes_200123 folder
addpath(genpath(pathtmp));
savedir = [pathtmp '\Dataset\Data\Fig4_generated_data'];

rand_layers_ind = [2, 6, 10, 12 14];    % Index of convolutional layer of AlexNet
number_sets = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]; % Candidiate numerosities of stimulus
LOI = 'relu5';  % Name of layer at which the activation will be measured
image_iter = 10;  % Number of images for a given condition
p_th1 = 0.01; p_th2 = 0.01; p_th3 = 0.01;  % Significance levels for two-way ANOVA
vis_N = 4;      % Number of example filters
PN_ex = [1, 5, 12, 16];  % Example PN
p_thtmp = 0.1; % Visualizing p value threshold
WV_eg = [1 0.75 0.5];
issavefig = 1;   % save fig files?

vis_L = rand_layers_ind(1); % visualizing layer
edgesh = -0.15:0.005:0.15; % bin edges of weight distribution
wrange = -0.15:0.0005:0.15; % range of weights

NDtmp = -30:30;

%% Loading pretrained network
% load('Alexnet_2018b.mat');    % load pretrained network
net = alexnet; % loading network from deep learning toolbox
% one can check the network architecture by using the command below
% analyzenetwork(net)

%% Generating stimulus set
if generatedat
    disp(['[Step 1/6] Generating a stimulus set...'])
    [image_sets_standard, image_sets_control1, image_sets_control2, polyxy]...
        = Stimulus_generation_Nasr(number_sets, image_iter);
    if savedat
        save([savedir '\1d_Stimulusset'], 'image_sets_standard', 'image_sets_control1', 'image_sets_control2', 'polyxy');
    end
else
    load([pathtmp '\Dataset\Data\1d_Stimulusset']);
end

%% WV simulation
if generatedat
    resp_tot_sigs = cell(iter, length(WV_eg));
    for ii = 1:length(WV_eg)
        for iterind = 1:iter
            
            %% Randomly permuting weights
            net_rand = Randomizeweightsig(net, rand_layers_ind, WV_eg(ii));
            
            %% Calculating response to stimulus
            disp(['[Step 2/6 of ' num2str(ii) '/' num2str(length(WV_eg)) 'th case] Calculating response to stimulus...'])
            response_tot_standard_RP = getactivation(net_rand, LOI, image_sets_standard);
            response_tot_control1_RP = getactivation(net_rand, LOI, image_sets_control1);
            response_tot_control2_RP = getactivation(net_rand, LOI, image_sets_control2);
            % get total response matrix
            response_tot_RP = cat(2,response_tot_standard_RP, response_tot_control1_RP, response_tot_control2_RP);
            units_N_RP = size(response_tot_RP, 3);
            
            %% Getting p-values of two-way ANOVA from response
            disp(['[Step 3/6 of ' num2str(ii) '/' num2str(length(WV_eg)) 'th case] Obtaining p-values for two-way ANOVA test...'])
            pvalues_RP = getpv(response_tot_RP);
            % pvalues2_RP = getpvforeach(response_tot_RP);
            
            %% Analyzing p-values to find number selective neurons
            disp(['[Step 4/6 of ' num2str(ii) '/' num2str(length(WV_eg)) 'th case] Analyzing p-values to find number selective neurons...'])
            pv1 = pvalues_RP(1,:); pv2 = pvalues_RP(2,:);pv3 = pvalues_RP(3,:);
            ind1 = (pv1<p_th1);ind2 = (pv2>p_th2);ind3 = (pv3>p_th2);
            ind_NS = find(ind1.*ind2.*ind3); % indices of number selective units
            
            %% Calculating mean response
            disp(['[Step 5/6 of ' num2str(ii) '/' num2str(length(WV_eg)) 'th case] Calculating average response...'])
            resp_mean_RP = squeeze((mean(response_tot_RP, 2))); resp_std_RP = squeeze(std(response_tot_RP, 0,2));
            resp_mean_standard_RP = squeeze((mean(response_tot_standard_RP, 2))); resp_std_standard = squeeze(std(response_tot_standard_RP, 0,2));
            resp_mean_control1_RP = squeeze((mean(response_tot_control1_RP, 2))); resp_std_control1 = squeeze(std(response_tot_control1_RP, 0,2));
            resp_mean_control2_RP = squeeze((mean(response_tot_control2_RP, 2))); resp_std_control2 = squeeze(std(response_tot_control2_RP, 0,2));
            
            %% Calculating preferred number of number neurons
            disp(['[Step 6/6 of ' num2str(ii) '/' num2str(length(WV_eg)) 'th case] Calculating the preferred numerosity...'])
            response_NS_tot_RP = response_tot_RP(:,:,ind_NS);
            response_NS_mean_RP = squeeze(mean(response_NS_tot_RP, 2));
            [M,PNind_RP] = max(response_NS_mean_RP);
            units_PN_RP = zeros(1,units_N_RP)/0; units_PN_RP(ind_NS) = PNind_RP; % preferred number for each neuron
            
            resp_tot_sigs{iterind, ii} = mean(response_NS_tot_RP,2);
        end
    end
else
    load('Data_Fig4_WV.mat')
end

%% Get sigma and R2, and response
resp_tot_sets = cell(1,length(WV_eg));

for sigind = 1:length(WV_eg)
    resptmpp = zeros(length(number_sets),1,0);
    
    for iterind = 1:iter
        resptmp = resp_tot_sigs{iterind, sigind};
        resptmpp = cat(3, resptmpp, resptmp);
    end
    resp_tot_sets{sigind} = resptmpp;
    
end
baselineresponse = resp_tot_sets{length(WV_eg)};
response_threshold = 0.05*mean(baselineresponse(:));

%% Average tuning curves
avCV_tot_sigs = zeros(iter, length(NDtmp), length(WV_eg));
for jj = 1:length(WV_eg)
    sigtmp = WV_eg(jj);
    for ii = 1:iter
        resptmp = resp_tot_sigs{ii, jj};
        indtmp = squeeze(mean(resptmp))>response_threshold;
        response_NS_meantmp = squeeze(mean(resptmp(:,:,indtmp), 2));
        [M,PNindtmp] =max(response_NS_meantmp);  [ND, respsets] = getAveTC(response_NS_meantmp, PNindtmp, number_sets);
        tmp = nanmean(respsets,1);    avCV_tot_sigs(ii,:,jj) = tmp;
    end
end

%% Figure 4a. Kernels of Alexnet
colortmp = { 'b', 'g', 'r'};

netseg = cell(1,length(WV_eg));
for ii = 1:length(WV_eg)
    WVtmp = WV_eg(ii);
    net_rand_tmp = Randomizeweightsig(net, rand_layers_ind, WVtmp);
    netseg{ii} = net_rand_tmp;
end

ff = figure; set(gcf,'Visible', 'off');
weight_tot = [];
for ind_tl = 1:length(rand_layers_ind)
    targetlayer_ind = rand_layers_ind(ind_tl);
    weight_conv = net.Layers(targetlayer_ind ,1).Weights;
    bias_conv = net.Layers(targetlayer_ind ,1).Bias;
    weight_tot = [weight_tot;weight_conv(:)];
end
subplot(6,4,[1 2 5 6])
histogram(weight_tot(:),edgesh, 'Normalization', 'probability')
set(gcf,'Position',[100 100 1000 400])
xlabel('Weight'); ylabel('Ratio'); ylim([0 0.2]); yticks([0 0.1 0.2]); xlim([-0.08 0.08]); xticks([-0.08 0 0.08])

for ii = 1:length(netseg)
    net_rand_tmp = netseg{ii};
    weight_tot = [];
    for ind_tl = 1:length(rand_layers_ind)
        targetlayer_ind = rand_layers_ind(ind_tl);
        weight_conv = net_rand_tmp.Layers(targetlayer_ind ,1).Weights;
        bias_conv = net_rand_tmp.Layers(targetlayer_ind ,1).Bias;
        weight_tot = [weight_tot;weight_conv(:)];
    end
    wstd = std(weight_tot(:)); wmean = mean(weight_tot(:));
    Gausstmp = exp(-(wrange-wmean).^2/(2*wstd^2)); Gausstmp = Gausstmp/max(Gausstmp);
    edgesh = -0.15:0.005:0.15; % bin edges of weight distribution
    subplot(6,4,[3 4 7 8])
    hold on;plot(wrange(1:end), Gausstmp(1:end), colortmp{ii})
end
legend({'100%', '75%', '50%'})
xlabel('Weight'); ylabel('Ratio'); ylim([0 1.2]); yticks([]); xlim([-0.08 0.08]); xticks([-0.08 0 0.08]);
% Visualizing example filter
for ii = 0:length(netseg)
    if ii==0
        nettmp = net;
    else
        nettmp = netseg{ii};
    end
    tmp = nettmp.Layers(vis_L).Weights;sztmp = size(tmp,4); indtmpp=randi(sztmp, [1,3]);
    for ind = 1:length(indtmpp)
        tmp2 = squeeze(tmp(:,:,1,indtmpp(ind)));
        if ii==0 && ind ==1
            caxtmp = max(abs(min(tmp2(:))), abs(max(tmp2(:))));
            
        end
        subplot(6,4,12+1+ii+(ind-1)*4); imagesc(squeeze(tmp(:,:,1,indtmpp(ind)))); axis image off; caxis([-caxtmp caxtmp])
        if ii ~=0
            if ind ==1;title(['\phi = ' num2str(125-25*ii) '%']);end
        end
    end
    colormap(gray);
end
if issavefig; savefig([pathtmp '\Figs\4a']); end
close(ff);

%% Figure. 4C: Average tuning curves
ff = figure;set(gcf,'Visible', 'off');
for ii = 1:length(WV_eg)
    sigind = ii; hold on
    tmp2 = squeeze(avCV_tot_sigs(:,:,sigind));
    tmp = nanmean(tmp2, 1); tmpstd = nanstd(tmp2, [],1);
    shadedErrorBar(ND(1:2:61), tmp(1:2:61), tmpstd(1:2:61), 'Lineprops', colortmp{ii})
    title('0.5 0.75 1')
end
xlim([-30 30]);ylim([0 1]);yticks([0 0.2 0.4 0.6 0.8 1])
xlabel('Numerical distance'); ylabel('Normalized response'); title('Average tuning curves')
if issavefig; savefig([pathtmp '\Figs\4c']); end
close(ff);

%% Visualizing figures
figure('WindowState', 'maximized')
weight_tot = [];
for ind_tl = 1:length(rand_layers_ind)
    targetlayer_ind = rand_layers_ind(ind_tl);
    weight_conv = net.Layers(targetlayer_ind ,1).Weights;
    bias_conv = net.Layers(targetlayer_ind ,1).Bias;
    weight_tot = [weight_tot;weight_conv(:)];
end

subplot(6,6,[1 2 7 8])
histogram(weight_tot(:),edgesh, 'Normalization', 'probability')
set(gcf,'Position',[100 100 1000 400])
xlabel('Weight'); ylabel('Ratio'); ylim([0 0.2]); yticks([0 0.1 0.2]); xlim([-0.08 0.08]); xticks([-0.08 0 0.08])
box off;
title('Fig 4a. Pre-trained kernels')
for ii = 1:length(netseg)
    net_rand_tmp = netseg{ii};
    weight_tot = [];
    for ind_tl = 1:length(rand_layers_ind)
        targetlayer_ind = rand_layers_ind(ind_tl);
        weight_conv = net_rand_tmp.Layers(targetlayer_ind ,1).Weights;
        bias_conv = net_rand_tmp.Layers(targetlayer_ind ,1).Bias;
        weight_tot = [weight_tot;weight_conv(:)];
    end
    wstd = std(weight_tot(:)); wmean = mean(weight_tot(:));
    Gausstmp = exp(-(wrange-wmean).^2/(2*wstd^2)); Gausstmp = Gausstmp/max(Gausstmp);
    edgesh = -0.15:0.005:0.15; % bin edges of weight distribution
    
    subplot(6,6, [3 4 9 10]); hold on;
    plot(wrange(1:end), Gausstmp(1:end), colortmp{ii});
    title('Controlled Gaussian model')
end
legend({'100%', '75%', '50%'})
xlabel('Weight'); ylim([0 1.2]); xlim([-0.08 0.08]); xticks([-0.08 0 0.08]); yticks([])

% Visualizing example filter
for ii = 0:length(netseg)
    if ii==0
        nettmp = net;
    else
        nettmp = netseg{ii};
    end
    tmp = nettmp.Layers(vis_L).Weights;sztmp = size(tmp,4); indtmpp=randi(sztmp, [1,3]);
    for ind = 1:length(indtmpp)
        tmp2 = squeeze(tmp(:,:,1,indtmpp(ind)));
        if ii==0 && ind ==1
            caxtmp = max(abs(min(tmp2(:))), abs(max(tmp2(:))));
            
        end
        if ii==0; indtmp = ii; else; indtmp = ii+1; end
        
        subplot(6,6,18+1+indtmp+(ind-1)*6);
        imagesc(squeeze(tmp(:,:,1,indtmpp(ind)))); axis image off; caxis([-caxtmp caxtmp])
        if ii ~=0
            if ind ==1;title(['\phi = ' num2str(125-25*ii) '%']);end
        else
            if ind ==1; title('Pre-trained kernels');end
        end
    end
    colormap(gray);
end

subplot(6,6, [5 6 11 12])
for ii = 1:length(WV_eg)
    sigind = ii; hold on
    tmp2 = squeeze(avCV_tot_sigs(:,:,sigind));
    tmp = nanmean(tmp2, 1); tmpstd = nanstd(tmp2, [],1);
    shadedErrorBar(ND(1:2:61), tmp(1:2:61), tmpstd(1:2:61), 'Lineprops', colortmp{ii})
    title('0.5 0.75 1')
end
xlim([-30 30]);ylim([0 1]);yticks([0 0.2 0.4 0.6 0.8 1])
xlabel('Numerical distance'); ylabel('Normalized response'); title('Fig 4c. Average tuning curves')
set(gca, 'xtick', -30:10:30, 'ytick', [0 1])
if issavefig; savefig([pathtmp '\Figs\4c']); end
