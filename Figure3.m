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

% This code performs a demo simulation for Fig. 3 in the manuscript.
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
savedir = [pathtmp '\Dataset\Data\Fig3_generated_data'];

%% Setting parameters
rand_layers_ind = [2, 6, 10, 12 14];    % Index of convolutional layer of AlexNet
number_sets = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]; % Candidiate numerosities of stimulus
image_iter = 10;  % Number of images for a given condition
p_th1 = 0.01; p_th2 = 0.01; p_th3 = 0.01;  % Significance levels for two-way ANOVA
layer_investigate = [4,5]; % Indices of convolutional layer to investigate
layer_last = layer_investigate(2);   %
array_sz = [55 55 96; 27 27 256; 13 13 384; 13 13 384; 13 13 256]; % Size of Alexnet
layers_name = {'relu1', 'relu2', 'relu3', 'relu4', 'relu5'};
LOI = 'relu4';

isrelativ = 1; %% Set mean of weights as zero
L4investigatePN = [1 16]; % Investigated PN in Conv4
L5investigatePN = 1:16; % Investigated PN in Conv5

issavefig = 1;   % Save fig files?
R2_th = 0.9; % threshold of R2 in visualization
egN = 10; % # of inc/dec units for visualization

col = [0 0/4470 0.7410;0.85 0.325 0.098;0.929 0.694 0.125;...
    0.494 0.184 0.556;0.466 0.674 0.188;0.301 0.745 0.933;0.635 0.0780 0.1840];
colortmp = cat(1, col, col, col, col); % color codes

%% Loading pretrained network
%load('Alexnet_2018b.mat');
net = alexnet; % Loading network from deep learning toolbox
% analyzenetwork(net); % Checking the network architecture

%% Generating stimulus set
if generatedat
    disp(['[Step 1/3] Generating a stimulus set...'])
    [image_sets_standard, image_sets_control1, image_sets_control2, polyxy]...
        = Stimulus_generation_Nasr(number_sets, image_iter);
    if savedat
        save([savedir '\1d_Stimulusset'], 'image_sets_standard', 'image_sets_control1', 'image_sets_control2', 'polyxy');
    end
else
    load([pathtmp '\Dataset\Data\1d_Stimulusset']);
end

%% Calculating response of L4
if generatedat
    [response_tot_blank, net_changed,response_NS_mean_RP, ...
        ind_NS, NS_ind, PNind_RP] = ...
        getIncdecunits(rand_layers_ind, LOI, net, p_th1, p_th2, p_th3, ...
        image_sets_standard, image_sets_control1, image_sets_control2);
else
    load([pathtmp '\Dataset\Data\3a_incdecinL4'])
end

%% Generating/analyzing backtracking data
if generatedat
    disp('[Step 2/3] Calculating preferred numerosity of Conv4 and Conv5 units...')
    
    generateBacktrackingdata(generatedat,rand_layers_ind...
        , number_sets, iter, net, layers_name, pathtmp, layer_investigate, ...
        p_th1, p_th2, p_th3, layer_last, array_sz, ...
        image_sets_standard, image_sets_control1, image_sets_control2)
    
    %% Analyzing backtracking data
    disp('[Step 3/3] Obtaining connection profiles...')
    
    [weightdist_foreachL5PN_tot, weightdist_control_tot, incdecportion_foreachL5PN_tot...
        , weightdist_foreachL5selective_tot, incdecportion_foreachL5selective_tot...
        , weightdist_foreachL5Nonselective_tot, incdecportion_foreachL5Nonselective_tot, ...
        incdecportion_foreachL5_tot] = analyzefig3data(iter,  ...
        pathtmp, L4investigatePN, L5investigatePN);
else
    %% Loading data
    load([pathtmp '\Dataset\Data\Data_Fig3_Backtracking'])
end

%% Calculating weights of increasing/decreasing units (in L4) connected to L5 neurons
if generatedat
    [meantmp, stdtmp] = getstatisticsofweightdist(net, rand_layers_ind(layer_last));
    [connectedweights, control1s, NS_wmean, NNS_wmean] = ...
        analyzeConnections(meantmp, L5investigatePN, L4investigatePN...
        , weightdist_foreachL5PN_tot, weightdist_control_tot, ...
        weightdist_foreachL5Nonselective_tot, iter, isrelativ);
    save([savedir '\3f_weightbias'], 'connectedweights', 'control1s', 'NS_wmean', 'NNS_wmean')
else
    load([pathtmp '\Dataset\Data\3f_weightbias'])
end

%% Figure 3a. Decreasing/increasing units
figure('units','normalized','outerposition',[0 0.25 1 0.5])

respblanktmp = squeeze(response_tot_blank(:,:,NS_ind))';
%size(response_NS_mean_RP)
for PNtmp = [1,16]
    if PNtmp ==1
        indtmp = (PNind_RP ==PNtmp); indtmpp = respblanktmp>0;
        %         indtmppp = R2s_L4>R2_th;
        indtmppp = ones(1,length(PNind_RP));
        %         indtmp1 = find(indtmp.*indtmpp.*indtmppp);
        indtmpppp = ((max(response_NS_mean_RP)-respblanktmp)<=0);
        indtmp1 = find(indtmp.*indtmpp.*indtmppp.*indtmpppp); indtmp2 = datasample(indtmp1,egN);
        subplot(2,6,1); hold on;
        for ii = 1:egN
            blkrsp = respblanktmp(indtmp2(ii));
            %             if isdiffcolor
            s = plot(number_sets, response_NS_mean_RP(:,indtmp2(ii))/blkrsp, 'Color', colortmp(ii,:));ylim([0 1]);
            s.Color(4) = 0.2;  hold on
            s = plot([-5 1], [1, response_NS_mean_RP(1,indtmp2(ii))/blkrsp], 'Color', colortmp(ii,:));
            s.Color(4) = 0.2;
            %             else
            %                 s = plot(number_sets, response_NS_mean_RP(:,indtmp2(ii))/blkrsp, 'r');ylim([0 1]);
            %                 s.Color(4) = 0.2;  hold on
            %                 s = plot([-5 1], [1, response_NS_mean_RP(1,indtmp2(ii))/blkrsp], 'r');
            %                 s.Color(4) = 0.2;
            %             end
        end
        indDec = logical(indtmp.*indtmpp.*indtmppp); resptmps = response_NS_mean_RP(:,indDec);
        blkrsps = respblanktmp(indDec); resptmpsmean = mean(resptmps, 2);
        resptmpsnormMean = resptmpsmean/mean(blkrsps);
        plot(number_sets, resptmpsnormMean, 'k'); s = plot([-5 1], [1, resptmpsnormMean(1)], 'k');
        title('Fig 3a. Decreasing units (Conv4)'); ylabel('Response (A.U.)')
        xticks([-5 0 10 20 30]); xticklabels({'blank', '0', '10', '20','30'})
        set(gca, 'ytick', [0 1])
    else
        subplot(2,6,7); hold on;
        indtmp = (PNind_RP==PNtmp);
        %indtmppp = R2s_L4>R2_th;
        indtmppp = ones(1,length(PNind_RP));
        indtmpp = respblanktmp==0;
        indtmp1 = find(indtmp.*indtmpp.*indtmppp);   indtmp2 = datasample(indtmp1,egN);
        for ii = 1:egN
            blkrsp = respblanktmp(indtmp2(ii));
            resptmp = response_NS_mean_RP(:,indtmp2(ii));  hold on;
            %             if isdiffcolor
            s = plot(number_sets, resptmp/max(resptmp), 'Color', colortmp(ii,:));   s.Color(4) = 0.2;
            s = plot([-5 1], [0 resptmp(1)/max(resptmp)], 'Color', colortmp(ii,:));ylim([0 1]);    s.Color(4) = 0.2;
            %             else
            %                 s = plot(number_sets, resptmp/max(resptmp), 'Color', 'b');   s.Color(4) = 0.2;
            %                 s = plot([-5 1], [0 resptmp(1)/max(resptmp)], 'Color', 'b');ylim([0 1]);    s.Color(4) = 0.2;
            %             end
        end
        indInc = logical(indtmp.*indtmpp.*indtmppp);
        resptmps = response_NS_mean_RP(:,indInc);
        blkrsps = respblanktmp(indInc);
        
        resptmpsmean = mean(resptmps, 2);
        resptmpsnormMean = resptmpsmean/max(resptmpsmean);
        plot(number_sets, resptmpsnormMean, 'k')
        s = plot([-5 1], [0, resptmpsnormMean(1)], 'k');
        title('Increasing units (Conv4)');xlabel('Numerosity');ylabel('Response (A.U.)')
        xticks([-5 0 10 20 30]); xticklabels({'blank', '0', '10', '20','30'})
        set(gca, 'ytick', [0 1])
    end
end

%% Figure 3f. dec, inc weight bias
check1std = squeeze(std(connectedweights));
diffcheck1 = squeeze(connectedweights(:,:,1)-connectedweights(:,:,2));
diffcontrol1 = squeeze(control1s(:,:,1)-control1s(:,:,2));
% Wilcoxon signed rank test
p1 = signrank(squeeze(connectedweights(:,3,1)), squeeze(connectedweights(:,3,2)), 'tail', 'right');
p2 = signrank(squeeze(connectedweights(:,13,1)), squeeze(connectedweights(:,13,2)), 'tail', 'left');

subplot(2,6,12); hold on;
per_shaded = fill([number_sets, fliplr(number_sets)], [mean(diffcheck1, 1)+std(diffcheck1, [],1), fliplr(mean(diffcheck1, 1)-std(diffcheck1, [],1))], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
perm_bias = plot(number_sets, mean(diffcheck1, 1), 'r');
% shadedErrorBar(number_sets, mean(diffcheck1, 1), std(diffcheck1, [],1))
%shadedErrorBar(number_sets, mean(diffcontrol1), std(diffcontrol1), 'Lineprops','r')

%% Summation model
if simulation_option == 1
    %% Setting parameters
    % Sigma of lognormal distribution sampled from the acitivity of decreasing (PN=1) and increasing (PN=30) units in the permuted AlexNet
    log_sig_dec_mean = 2.49;    % Mean of sigma sampled from decreasing units
    log_sig_dec_std = 3.76;     % STD of sigma sampled from decreasing units
    log_sig_inc_mean = 2.03;    % for increasing units
    log_sig_inc_std = 1.95;
    
    % Distribution of convolutional weights
    mean_weight = -6.25*10^(-4);      % Mean of weights
    std_weight = 4.70*10^(-3);         % STD of weights; scaled considering the distribution of weights in the pre-trained AlexNet: avg/std = -0.132)
    
    % List of tested numerosities
    num_list = [1 2:2:30];
    num_list_log = log2(num_list);
    num_list_ext = [num_list, 0];
    
    % Network specification
    nunit = 60;                     % Number of decreasing and increasing units
    nneuron = 10000;                 % Number of output neurons
    ntrial = 1;                      % Number of simulations
    
    bb_weight = linspace(-0.03, 0.03, 61);    % Bins for histogram of weights
    bb_sigma = linspace(-10, 20, 101);
    
    xx_axis = -5:35;
    
    color_dec = [237 30 121]/255;
    color_inc = [0 113 188]/255;
    
    %% Variables for saving simulation results
    w_dec_save = zeros(ntrial, nunit, nneuron);
    w_inc_save = zeros(ntrial, nunit, nneuron);
    
    tun_dec = zeros(ntrial, length(num_list), nunit);    % Tuning curves of each unit
    tun_inc = zeros(ntrial, length(num_list), nunit);
    
    tun_all = zeros(length(num_list), ntrial, length(num_list));
    tun_all_save = zeros(ntrial, length(num_list), nneuron);
    pn_all_save = zeros(ntrial, nneuron);
    
    pn_all = zeros(ntrial, length(num_list));
    sigma_all = zeros(ntrial, length(num_list));
    sigma_log_all = zeros(ntrial, length(num_list));
    
    w_dec_mean = zeros(ntrial, length(num_list));
    w_inc_mean = zeros(ntrial, length(num_list));
    
    sigma_Weber = zeros(size(num_list));
    sigma_Weber_log = zeros(size(num_list));
    
    %% Generating tuning curves
    log_sig_dec = randn(nunit, ntrial)*log_sig_dec_std + log_sig_dec_mean;  % Randomly generated sigma of tuning curves
    log_sig_inc = randn(nunit, ntrial)*log_sig_inc_std + log_sig_inc_mean;
    
    for pp = 1:ntrial
        for ii = 1:nunit
            tun_dec(pp, :, ii) = exp(-((num_list_log-log2(1)).^2)/(2*(log_sig_dec(ii, pp)).^2));
            tun_inc(pp, :, ii) = exp(-((num_list_log-log2(30)).^2)/(2*(log_sig_inc(ii, pp)).^2));
        end
    end
    
    %% Weighted summation
    for pp = 1:ntrial
        w_dec = zeros(nunit, nneuron);
        w_inc = zeros(nunit, nneuron);
        
        tun_map = zeros(length(num_list), nneuron);
        pn_map = zeros(nneuron, 1);
        sigma_map = zeros(nneuron, 1);
        sigma_log_map = zeros(nneuron, 1);
        
        for ii = 1:nneuron
            w_dec(:, ii) = mean_weight + randn(nunit, 1)*std_weight;
            w_inc(:, ii) = mean_weight + randn(nunit, 1)*std_weight;
        end
        
        res_1_temp = squeeze(tun_dec(pp,:,:))*squeeze(w_dec(:, :));
        res_30_temp = squeeze(tun_inc(pp,:,:))*squeeze(w_inc(:, :));
        
        res_temp =  res_1_temp + res_30_temp;
        
        res_temp(res_temp<0) = 0;
        
        tun_map(:, :) = res_temp;
        [val_temp, ind_temp] = max(res_temp);
        ind_temp(sum(res_temp) == 0) = length(num_list_ext);
        pn_map(:) = num_list_ext(ind_temp);
        
        bb = [0 1 2:2:30];
        pn_dist = zeros(length(num_list), 1);
        for ii = 1:length(num_list)
            pn_dist(ii) = sum(pn_map(:)==num_list(ii));
        end
        pn_dist = pn_dist/sum(pn_dist);
        pn_all(pp, :) = pn_dist;
        
        for ii = 1:length(num_list)
            tun_all(ii, pp, :) = mean(tun_map(:, pn_map == num_list(ii)), 2);
        end
        
        tun_all_save(pp, :, :) = tun_map;
        pn_all_save(pp, :, :) = pn_map;
        
        %% Average tuning curves
        tun_temp = reshape(tun_map, [length(num_list), nneuron]);
        [max_val, max_ind] = max(tun_temp, [], 1);
        max_ind(max_val==0) = nan;
        
        sigma_norm = nan(length(num_list), 1);
        sigma_norm_log = nan(length(num_list), 1);
        
        xtmp = num_list;
        xtmp_log = num_list_log;
        options = fitoptions('gauss1', 'Lower', [0 0 0], 'Upper', [1.5 30 100]);
        
        for ii = 1:length(num_list)
            sel_ind = find(max_ind == ii);
            temp = nanmean(tun_temp(:, sel_ind), 2);
            temp = (temp-min(temp))/(max(temp)-min(temp));
            
            xtmp = num_list;
            if ~isnan(temp)
                [f, gof] = fit(xtmp.', temp, 'gauss1', options);
                
                sigma_norm(ii) = f.c1/sqrt(2);
                
                xtmp = num_list_log;
                [f_log, gof_log] = fit(xtmp.', temp, 'gauss1', options);
                
                sigma_norm_log(ii) = f_log.c1/sqrt(2);
            end
        end
        
        sigma_all(pp, :) = sigma_norm;
        sigma_log_all(pp, :) = sigma_norm_log;
        
        %% Weight bias
        for ii = 1:length(num_list)
            w_dec_temp = (w_dec(:, (pn_map == num_list(ii))) - mean(w_dec(:)))/std(w_dec(:));
            w_inc_temp = (w_inc(:, (pn_map == num_list(ii))) - mean(w_inc(:)))/std(w_inc(:));
            
            w_dec_mean(pp, ii) = mean(w_dec_temp(:));
            w_inc_mean(pp, ii) = mean(w_inc_temp(:));
        end
        
        w_dec_save(pp, :, :) = w_dec;
        w_inc_save(pp, :, :) = w_inc;
    end
elseif simulation_option == 2
    load([pathtmp '\Dataset\Data\3bf_summation_model'])
end

trial_ind = 1;
num_shown_tun = 20;

tun_single_trial = squeeze(tun_all_save(trial_ind, :, :));
pn_single_trial = squeeze(pn_all_save(trial_ind, :, :));

% Examples of decreasing tuning curves
subplot(2,6,2); hold on;
plot(squeeze(tun_dec(1,:,randperm(nunit, num_shown_tun))), 'color', color_dec);
title('Fig 3b. Model curves (Dec.)')
ylabel('Response (A.U.)')
set(gca, 'xtick', [1 6 11 16], 'xticklabel', [1 10 20 30], 'ytick', [0 1])

% Examples of increasing tuning curves
subplot(2,6,8); hold on;
plot(squeeze(tun_inc(1,:,randperm(nunit, num_shown_tun))), 'color', color_inc);
title('Model curves (Inc.)')
xlabel('Numerosity')
ylabel('Response (A.U.)')
set(gca, 'xtick', [1 6 11 16], 'xticklabel', [1 10 20 30], 'ytick', [0 1])

% Example tuning curves
subplot_ind = [5:8, 17:20, 29:32, 41:44];
xtick_ind = [41:44];
ytick_ind = [5 17 29 41];
for ii = 1:length(num_list)
    tun_each_num = tun_single_trial(:, pn_single_trial==num_list(ii));
    norm_tun_each_num = tun_each_num./repmat(max(tun_each_num), [length(num_list), 1]);
    
    subplot(4,12,subplot_ind(ii)); hold on;
    plot(norm_tun_each_num, 'color', [0.8 0.8 0.8 0.5])
    plot(mean(norm_tun_each_num,2), 'k')
    set(gca, 'xtick', [], 'ytick', [])
    if ismember(subplot_ind(ii), xtick_ind); set(gca, 'xtick', [1 6 11 16], 'xticklabel', [1 10 20 30]); end
    if ismember(subplot_ind(ii), ytick_ind); set(gca, 'ytick', [0 1]); end
    if ismember(subplot_ind(ii), xtick_ind) && ismember(subplot_ind(ii), ytick_ind); xlabel('Numerosity'); ylabel('Normalized response'); end
    title(['PN = ' num2str(num_list(ii))])
    if ii == 1; title(['Weighted sum: PN = ' num2str(num_list(ii))]); end
end

% Distribution of preferred numerosity
subplot(2,6,5); hold on;
pn_dist = zeros(length(num_list), 1);
for ii = 1:length(num_list)
    pn_dist(ii) = mean(pn_all(:, ii));
end
plot(num_list, pn_dist, 'k')

load('PN_distribution_from_monkey_Nieder_2007.mat', 'PN_monkey')   % Loding monkey data from Nieder et al., 2007
plot(num_list, PN_monkey, 'color', [0 0.7 0])
set(gca, 'xtick', 0:10:30, 'ytick', 0:0.05:0.35, 'yticklabel', {[0], [], [0.1], [], [0.2], [], [0.3], []})
title('Fig 3c. PN distribution')
legend({'Summation model', 'Monkey data'})
legend boxoff
xlabel('Preferred numerosity')
ylabel('Probability')
xlim([0 30]); ylim([0 0.35])

% Weber-Fechner law
subplot(2,6,6); hold on;
sigma_model_linear = nanmean(sigma_all,1);
sigma_model_log = nanmean(sigma_log_all,1);

load('sigma_from_monkey_Nieder_2007.mat', 'sig_monkey_linear', 'sig_monkey_log')       % Loding monkey data from Nieder et al., 2007

scatter(num_list, sig_monkey_linear, 'MarkerEdgeColor', color_linear, 'MarkerFaceColor', 'none')
scatter(num_list, sig_monkey_log, 'MarkerEdgeColor', color_log, 'MarkerFaceColor', 'none')

scatter(num_list, sigma_model_linear, 'MarkerEdgeColor', color_linear, 'MarkerFaceColor', color_linear);
scatter(num_list, sigma_model_log, 'MarkerEdgeColor', color_log, 'MarkerFaceColor', color_log);

scatter(-0.5, 15, 20, 'o', 'MarkerEdgeColor', color_linear, 'MarkerFaceColor', color_linear)
text(1, 15, 'Linear', 'fontsize', 8)
scatter(-0.5, 13.3, 20, 'o', 'MarkerEdgeColor', color_log, 'MarkerFaceColor', color_log)
text(1, 13.3, 'Log', 'fontsize', 8)
scatter(-0.5, 11.7, 20, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k')
text(1, 11.7, 'Model', 'fontsize', 8)
scatter(-0.5, 10.1, 20, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'none')
text(1, 10.1, 'Monkey', 'fontsize', 8)

fit_Weber_linear = fit(num_list.', sigma_model_linear.', 'poly1');
fit_Weber_log = fit(num_list.', sigma_model_log.', 'poly1');

plot(xx_axis, fit_Weber_linear.p1*xx_axis + fit_Weber_linear.p2, 'color', color_linear)
plot(xx_axis, fit_Weber_log.p1*xx_axis + fit_Weber_log.p2, 'color', color_log)
title('Fig 3d. Weber-Fechner law')
xlabel('Preferred numerosity')
ylabel('\sigma of Gaussian fit')
xlim([-2 32]); ylim([-2 16])
set(gca, 'xtick', 0:10:30, 'ytick', 0:8:16)

% Calculating weight bias across numerosities
subplot(2,6,12);
if simulation_option == 1
    weight_diff = zeros(ntrial, length(num_list));
    for pp = 1:ntrial
        w_dec_temp = squeeze(w_dec_save(pp,:,:));
        w_inc_temp = squeeze(w_inc_save(pp,:,:));
        pn_temp = squeeze(pn_all_save(pp,:,:));
        
        for ii = 1:length(num_list)
            w_dec_temp2 = w_dec_temp(:, (pn_temp == num_list(ii)));
            w_inc_temp2 = w_inc_temp(:, (pn_temp == num_list(ii)));
            
            weight_diff(pp,ii) = nanmean(w_dec_temp2(:)) - nanmean(w_inc_temp2(:));
        end
    end
end

model_shaded = fill([num_list, fliplr(num_list)], [nanmean(weight_diff,1)+nanstd(weight_diff,0,1), fliplr(nanmean(weight_diff,1)-nanstd(weight_diff,0,1))], 'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
model_bias = plot(num_list, nanmean(weight_diff,1), 'k');
plot([0 num_list(end)], [0 0], 'k--')
xlim([0 30]); ylim([-1 1]*(2)*(10^(-3)));
title('Fig 3f. Input bias')
xlabel('Preferred numerosity')
ylabel('W_{Dec} - W_{Inc}')
legend([perm_bias, model_bias], {'Permuted', 'Summation model'})
legend boxoff
if issavefig; savefig([pathtmp '\Figs\3']); end
