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

% This code performs a demo simulation for Fig. 2 in the manuscript.
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
pathtmp = pwd;
addpath(genpath(pathtmp));
savedir = [pathtmp '\Dataset\Data\Fig2_generated_data'];

%% Setting parameters

rand_layers_ind = [2, 6, 10, 12 14];    % Index of convolutional layer of AlexNet
number_sets = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]; % Candidiate numerosities of stimulus
LOI = 'relu5';  % Name of layer at which the activation will be measured
image_iter = 10;  % Number of images for a given condition, image_iter = 50 was used for original simulation
p_th1 = 0.01; p_th2 = 0.01; p_th3 = 0.01;  % Significance levels for two-way ANOVA
PN_ex = [1, 5, 12, 16];  % Visualizing PN eg
p_thtmp = 0.1; % Visualizing p value threshold
layers_ind = [2,14]; % indices of Conv1 and Conv5
vis_N = 3;   % Number of example filters

issavefig = 1;   % save fig files?

% Numerosity comparison task parameters

iterforeachN = 20; % # of training data for each numerosity
iterforeachN_val = 20; % # of validation data for each numerosity
setNo = 1:3; % included sets in numerosity comparison task
sampling_numbers = [16]; % # of sampling neurons in comparison task
averageiter = 10; % # of sampling iterations
ND = -30:30; % numerical distance

%% Loading pretrained networks
% load('Alexnet_2018b.mat');    % load pretrained network
net = alexnet;

%% Generating stimulus set
if generatedat
    disp(['[Step 1/7] Generating a stimulus set...'])
    [image_sets_standard, image_sets_control1, image_sets_control2, polyxy]...
        = Stimulus_generation_Nasr(number_sets, image_iter);
    if savedat
        save([savedir '\1d_Stimulusset'], 'image_sets_standard', 'image_sets_control1', 'image_sets_control2', 'polyxy');
    end
else
    load([pathtmp '\Dataset\Data\1d_Stimulusset']);
end

%% Finding number selective neurons in permuted Alexnet
if generatedat
    response_NS_sets = cell(iter, 2);
    ind_NS_sets = cell(iter, 2); %% 2D: dim1 : iteration, dim2: pretrained/permuted
    
    for iterind = 1:iter
        %% Randomly permuting weights
        net_rand = Randomizeweight_permute(net, rand_layers_ind);
        
        %% Figure 2a. Visualizing permuted kernels
        ff = figure; set(gcf,'Visible', 'off')
        for layers = layers_ind
            tmp = net.Layers(layers).Weights;sztmp = 1:size(tmp,4);indtmpp=datasample(sztmp, vis_N,'Replace', false);
            for ind = 1:length(indtmpp)
                tmp2 = squeeze(tmp(:,:,1,indtmpp(ind)));caxtmp = max(abs(min(tmp2(:))), abs(max(tmp2(:))));
                subplot(3,6,ind+12*(floor(layers/10)));imagesc(squeeze(tmp(:,:,1,indtmpp(ind))));axis image off;caxis([-caxtmp caxtmp])
            end
            tmp = net_rand.Layers(layers).Weights;sztmp = 1:size(tmp,4);indtmpp=datasample(sztmp, vis_N,'Replace', false);
            for ind = 1:length(indtmpp)
                tmp2 = squeeze(tmp(:,:,1,indtmpp(ind)));caxtmp = max(abs(min(tmp2(:))), abs(max(tmp2(:))));
                ax=subplot(3,6,ind+3+12*(floor(layers/10)));imagesc(squeeze(tmp(:,:,1,indtmpp(ind))));axis image off;caxis(ax, [-caxtmp caxtmp])
            end
            colormap(gray);
        end
        sgtitle('Fig. 2a: Pretrained weights ------> permuted weights')
        if issavefig; savefig([pathtmp '\Figs\2a']); end
        close(ff)
        
        %% Figure 2b. Classification performance, 100 iteration
        load('IMAGENET_ALEXNET_for_test.mat')
        top1_accu_pret = squeeze(accuracy_tot(:, 1, 1)); top1_accu_perm = squeeze(accuracy_tot(:, 2, 1));
        ff = figure; set(gcf,'Visible', 'off');
        b1 = bar([1], [mean(top1_accu_pret)], 'FaceColor', [0.5 0.5 0.5]);ylim([0 0.8])
        x_min = std(top1_accu_pret); x_max = std(top1_accu_pret);
        hold on;errorbar(1,mean(top1_accu_pret), x_min, x_max, 'k')
        b2 = bar(2, mean(top1_accu_perm), 'r');
        x_min = std(top1_accu_perm); x_max = std(top1_accu_perm);
        hold on;errorbar(2, mean(top1_accu_perm), x_min, x_max, 'k')
        errorbarlogy; ylim([0.0001 1]); hold on; plot([0, 3], [0.001 0.001])
        ptmp = ranksum(top1_accu_pret, top1_accu_perm, 'tail', 'right');
        title('Fig. 2b: Image classification performance'); legend([b1 b2], {'pretrained', 'permuted'}); set(gca, 'xtick', [])
        if issavefig; savefig([pathtmp '\Figs\2b']); end
        close(ff)
        
        %% Calculating response to stimulus
        disp(['[Step 2/7] Calculating response to stimulus...'])
        response_tot_standard_RP = getactivation(net_rand, LOI, image_sets_standard);
        response_tot_control1_RP = getactivation(net_rand, LOI, image_sets_control1);
        response_tot_control2_RP = getactivation(net_rand, LOI, image_sets_control2);
        % get total response matrix
        response_tot_RP = cat(2,response_tot_standard_RP, response_tot_control1_RP, response_tot_control2_RP);
        units_N_RP = size(response_tot_RP, 3);
        
        %% Getting p-values of two-way ANOVA from response
        disp(['[Step 3/7] Obtaining p-values for two-way ANOVA test...'])
        pvalues_RP = getpv(response_tot_RP);
        % pvalues2_RP = getpvforeach(response_tot_RP);
        
        %% Analyzing p-values to find number selective neurons
        disp(['[Step 4/7] Analyzing p-values to find number selective neurons...'])
        pv1 = pvalues_RP(1,:); pv2 = pvalues_RP(2,:);pv3 = pvalues_RP(3,:);
        ind1 = (pv1<p_th1);ind2 = (pv2>p_th2);ind3 = (pv3>p_th2);
        ind_NS = find(ind1.*ind2.*ind3); % indices of number selective units
        
        %% Calculating mean response
        disp(['[Step 5/7] Calculating average response...'])
        resp_mean_RP = squeeze((mean(response_tot_RP, 2))); resp_std_RP = squeeze(std(response_tot_RP, 0,2));
        resp_mean_standard_RP = squeeze((mean(response_tot_standard_RP, 2))); resp_std_standard = squeeze(std(response_tot_standard_RP, 0,2));
        resp_mean_control1_RP = squeeze((mean(response_tot_control1_RP, 2))); resp_std_control1 = squeeze(std(response_tot_control1_RP, 0,2));
        resp_mean_control2_RP = squeeze((mean(response_tot_control2_RP, 2))); resp_std_control2 = squeeze(std(response_tot_control2_RP, 0,2));
        
        %% Calculating preferred number of number neurons
        disp(['[Step 6/7] Calculating the preferred numerosity...'])
        response_NS_tot_RP = response_tot_RP(:,:,ind_NS);
        response_NS_mean_RP = squeeze(mean(response_NS_tot_RP, 2));
        [M,PNind_RP] = max(response_NS_mean_RP);
        units_PN_RP = zeros(1,units_N_RP)/0; units_PN_RP(ind_NS) = PNind_RP; % preferred number for each neuron
        tmp1 = response_NS_tot_RP(:,1:image_iter, :); tmp1 = (mean(tmp1, 2));
        tmp2 = response_NS_tot_RP(:,image_iter+1:image_iter*2, :);tmp2 = (mean(tmp2, 2));
        tmp3 = response_NS_tot_RP(:,image_iter*2+1:image_iter*3,:);tmp3 = (mean(tmp3, 2));
        response_NS_sep_RP = cat(2,tmp1, tmp2, tmp3);
        
        %% Getting example tuning curves for individual neurons
        %pv4 = pvalues2_RP(2,:); pv5 = pvalues2_RP(1,:); pv6 = pvalues2_RP(3,:);
        %ind4 = pv4<p_thtmp; ind5 = pv5<p_thtmp; ind6 = pv6<p_thtmp;
        %isNS2 = (ind1.*ind2.*ind3.*ind4.*ind5.*ind6);
        resp_totmean_tmp = zeros(length(number_sets), length(PN_ex)); resp_totstd_tmp = zeros(length(number_sets), length(PN_ex));
        resp_standard_tmp = zeros(length(number_sets), length(PN_ex)); resp_ctr1_tmp = zeros(length(number_sets), length(PN_ex));
        resp_ctr2_tmp = zeros(length(number_sets), length(PN_ex));
        indseg = zeros(1,length(number_sets));
        for ii = 1:length(number_sets)
            PNtmp = ii;
            %     indcand = find(units_PN_RP ==PNtmp & isNS2);
            indcand = find(units_PN_RP ==PNtmp);
            if length(indcand)>0
                indcand2 = datasample(indcand, 1);
                resp_totmean_tmp(:,ii) = resp_mean_RP(:, indcand2);
                resp_totstd_tmp(:,ii) = resp_std_RP(:, indcand2)/sqrt(3*image_iter);
                resp_standard_tmp(:,ii) = resp_mean_standard_RP(:, indcand2);
                resp_ctr1_tmp(:,ii) = resp_mean_control1_RP(:, indcand2);
                resp_ctr2_tmp(:,ii) = resp_mean_control2_RP(:, indcand2);
                indseg(ii) = indcand2;
            end
        end
        if savedat
            save([savedir '\2c_Exampleresponses'], 'resp_totmean_tmp', 'resp_totstd_tmp', ...
                'resp_standard_tmp', 'resp_ctr1_tmp', 'resp_ctr2_tmp', 'indseg' );
        end
        
        %% Figure 2c. Number neurons in pre-trained AlexNet
        ff = figure; set(gcf,'Visible', 'off')
        sgtitle('Fig. 2c: Number neurons in permuted AlexNet')
        for ii = 1:length(PN_ex)
            subplot(2,2,ii)
            hold on
            shadedErrorBar(number_sets, resp_totmean_tmp(:,PN_ex(ii)), resp_totstd_tmp(:,PN_ex(ii)))
            plot(number_sets, resp_standard_tmp(:,PN_ex(ii)), 'b' )
            plot(number_sets, resp_ctr1_tmp(:,PN_ex(ii)), 'r' )
            plot(number_sets, resp_ctr2_tmp(:,PN_ex(ii)), 'g')
            % legend({'tot','std','ctr1', 'ctr2'});% scatter(number_sets, resp_mean(:,randi(43264)))
            
        end
        if issavefig; savefig([pathtmp '\Figs\2c']); end
        close(ff)
        
        response_NS_sets{iterind, 2} = response_NS_sep_RP;
        ind_NS_sets{iterind,2} = units_PN_RP;
        
    end
else
    load([pathtmp '\Dataset\Data\2c_Exampleresponses']);
    % load data for pretrained and permuted network, 100 iterations
    load('Data_NSsimulationfor100iterations')
end

%% PN distribution
PN_permutedraw = []; PN_pretrainedraw = [];
PNdist_perm_tot = zeros(iter, 16); PNdist_pret_tot = zeros(iter, 16);
for ii = 1:iter
    resp_NS_mean_pret = []; resp_NS_mean_perm = [];
    prettmp = response_NS_sets{ii,1}; permtmp = response_NS_sets{ii,2};
    response_NS_meantmp_pret = squeeze(mean(prettmp,2));
    response_NS_meantmp_perm = squeeze(mean(permtmp,2));
    resp_NS_mean_pret = cat(2, resp_NS_mean_pret, response_NS_meantmp_pret);
    resp_NS_mean_perm = cat(2, resp_NS_mean_perm, response_NS_meantmp_perm);
    [M,PNind_pret] =max(resp_NS_mean_pret);
    [M,PNind_perm] =max(resp_NS_mean_perm);
    PN_permutedraw = [PN_permutedraw, PNind_perm];
    PN_pretrainedraw = [PN_pretrainedraw, PNind_pret];
    for jj = 1:length(number_sets)
        PNdist_perm_tot(ii,jj) = sum(PNind_perm==jj)/length(PNind_perm);
        PNdist_pret_tot(ii,jj) = sum(PNind_pret==jj)/length(PNind_perm);
    end
end

load('PN_distribution_from_monkey_Nieder_2007.mat')
ff = figure; set(gcf,'Visible', 'off')
b1 = plot(number_sets, PN_monkey); alpha (0.3); hold on;
b2 = shadedErrorBar(number_sets, mean(PNdist_perm_tot,1), std(PNdist_perm_tot,[],1)); alpha (0.3);%b2.EdgeColor = 'none'; %/sum(PN_popul_RP));
% b3 = plot(number_sets, PN_pretrained); alpha (0.3)
xlabel('numerosity');ylabel('portion')
title('Fig. 2d: Preferred numerosity distribution')
legend([b1, b2.mainLine], {'monkey', 'permuted'})
if issavefig; savefig([pathtmp '\Figs\2d']); end
close(ff)

%% Analyzing data - Average tuning curves and tuning width in permuted Alexnet
if generatedat
    [tuning_curve_ave_per, sig_lin_per_tot, R2_lin_per_tot, ...
        sig_log_per_tot, R2_log_per_tot ]...
        = analyzefig2data(iter, number_sets, response_NS_sets, ind_NS_sets);
    if savedat
        save([savedir '\2ef_averagetuningcurves'], 'tuning_curve_ave_per', 'sig_lin_per_tot', 'R2_lin_per_tot', ...
            'sig_log_per_tot', 'R2_log_per_tot');
    end
else
    load([pathtmp '\Dataset\Data\2ef_averagetuningcurves'])
end

%% figure 2e. Average tuning curve
col = [0 0/4470 0.7410;0.85 0.325 0.098;0.929 0.694 0.125;0.494 0.184 0.556;0.466 0.674 0.188;0.301 0.745 0.933;0.635 0.0780 0.1840];
colortmp = cat(1, col, col, col, col);
options = fitoptions('gauss1', 'Lower', [0 0 0], 'Upper', [1.5 30 100]);
figure; set(gcf,'Visible', 'off'); hold on
xtmp = number_sets;sigmas = zeros(1,length(number_sets)); R2 = zeros(1,length(number_sets)); TNtmp = tuning_curve_ave_per;
for ii = 1:length(number_sets)
    hh=plot(number_sets, TNtmp(ii,:), 'Color', colortmp(ii,:));
    ytmp = TNtmp(ii,:);
    if isnan(ytmp)
        sigmas(ii) = nan;
    else
        %         f = fit(xtmp.', ytmp.', 'gauss1', options);
        %         sigmas(ii) = f.c1/2; xfinetmp = 1:0.01:30;
        %         tmp = f.a1*exp(-((xfinetmp-f.b1)/f.c1).^2);
        %         ymean = nanmean(ytmp);
        %     SStot = sum((ytmp-ymean).^2); %     SSres = sum((ytmp-tmp).^2); %     R2(ii) = 1-SSres./SStot;
    end
    %     hold on; h=plot(xfinetmp, tmp, 'Color', colortmp(ii,:));legend off;
    %     yysig = 0:0.5/16:0.5;
    %     plot([f.b1, f.b1+sigmas(ii)], [yysig(ii+1), yysig(ii+1)], 'Color', colortmp(ii,:))
end
xlabel('Numerosity');ylabel('Normalized response'); title('Fig. 2e: Average tuning curve')
if issavefig; savefig([pathtmp '\Figs\2e']); end

%% figure 2f. Weber-Fechner law
% load([pathtmp '\Dataset\Data\1fg_averagetuningcurves']) % tuning width data for pretrained network
load('sigma_from_monkey_Nieder_2007.mat')       % Loding monkey data from Nieder et al., 2007

figure; set(gcf,'Visible', 'off'); hold on
sig_lin_per = mean(sig_lin_per_tot, 1);
sig_log_per = mean(sig_log_per_tot, 1);
stdlin = std(sig_lin_per_tot, [],1);
stdlog = std(sig_log_per_tot, [],1);
s1 = scatter(number_sets, sig_lin_per, 'k', 'fill');
p1 = polyfit(number_sets, sig_lin_per, 1);
%errorbar(number_sets, sig_lin_per, stdlin, 'LineStyle', 'none');
plot(number_sets, p1(1)*number_sets+p1(2), 'k')
s2 = scatter(number_sets, sig_log_per, 'r', 'fill');
%errorbar(number_sets, sig_log_per, stdlog, 'LineStyle', 'none')
p2 = polyfit(number_sets, sig_log_per, 1);
plot(number_sets, p2(1)*number_sets+p2(2), 'r')
xlabel('Numerosity');ylabel('Sigma of Gaussian fit');ylim([0 16])
[r1,pv1] = corrcoef(number_sets, sig_lin_per);
[r2,pv2] = corrcoef(number_sets, sig_log_per);

s3 = scatter(x_axis, sig_monkey_linear, 'k'); s4 = scatter(x_axis, sig_monkey_log, 'r'); title('Fig. 2f: Weber-Fechner law')
legend([s1, s2, s3, s4], {'Permuted (lin)', 'Permuted (log)', 'Monkey (lin)', 'Monkey (log)'})
if issavefig; savefig([pathtmp '\Figs\2g']); end

%% Generating/loading data for numerosity comparison task
if generatedat
    disp(['[Step 7/7] Performing comparison task...'])
    generateComparisontaskdata(iter, pathtmp, generatedat, net, rand_layers_ind, ...
        iterforeachN, iterforeachN_val, image_iter, setNo, LOI, savedir, p_th1, p_th2, p_th3, ...
        image_sets_standard, image_sets_control1,image_sets_control2, number_sets)
    
    %% Analyzing data for numerosity comparison task
    [performance_pret_perm_sets, correctness_inputnumerosity_dat, correct_sets, wrong_sets] ...
        = analyzeComparisontaskdata(iter, pathtmp, generatedat, ...
        sampling_numbers, averageiter, ND, number_sets, savedir);
end

% loading
if ~ generatedat
    load('2h_performancematrix'); load('2i_correctwrong');
end

%% Figure 2h. Task performance
figure; set(gcf,'Visible', 'off')
tmp1 = (performance_pret_perm_sets(:,1)); % performance of pre-trained network
tmp2 = (performance_pret_perm_sets(:,2)); % performance of permuted network
tmp3 = (performance_pret_perm_sets(:,3)); % shuffled response
tmp4 = (performance_pret_perm_sets(:,4)); % shuffled response
b1=bar([1], [mean(tmp1)], 'k');hold on
b2=bar([2], [mean(tmp2)], 'r');
b3=bar([3], [mean(tmp4)], 'w');
errorbar([1,2,3], [mean(tmp1), mean(tmp2), mean(tmp4)], [std(tmp1), std(tmp2), std(tmp4)], 'b','LineStyle', 'none');
ylim([0.4 0.8]); set(gca, 'xtick', []); ylabel('Performance (%)')
legend([b1, b2, b3], {'Pre-trained','Permuted','Response shuffled'})
ptmp1 = ranksum(tmp1, tmp3, 'tail', 'right');
ptmp3 = ranksum(tmp1, tmp2, 'tail', 'left');
title('Fig. 2h: Task performance');
if issavefig; savefig([pathtmp '\Figs\2h']); end

%% Figure 2i. Number neurons response
figure; set(gcf,'Visible', 'off')
tmp1 = nanmean(correct_sets, 1); std1 = nanstd(correct_sets, [],1);
shadedErrorBar(ND(1:2:61), tmp1(1:2:61), std1(1:2:61), 'Lineprops', 'r');hold on
tmp2 = nanmean(wrong_sets, 1); std2 = nanstd(wrong_sets, [],1);
shadedErrorBar(ND(1:2:61), tmp2(1:2:61),std2(1:2:61), 'Lineprops', 'k');
xlabel('Numerical distance');ylabel('Normalized response');
ptmp = ranksum(correct_sets(:, 31), wrong_sets(:, 31), 'tail', 'right');
title('Fig. 2i: Number neurons response')
if issavefig; savefig([pathtmp '\Figs\2i']); end

%% Visualizing figures
figure;
net_rand = Randomizeweight_permute(net, rand_layers_ind);
for layers = layers_ind
    tmp = net.Layers(layers).Weights;sztmp = 1:size(tmp,4);indtmpp=datasample(sztmp, vis_N,'Replace', false);
    for ind = 1:length(indtmpp)
        tmp2 = squeeze(tmp(:,:,1,indtmpp(ind)));caxtmp = max(abs(min(tmp2(:))), abs(max(tmp2(:))));
        subplot(3,10,ind+20*(floor(layers/10)));
        imagesc(squeeze(tmp(:,:,1,indtmpp(ind))));
        axis image; set(gca, 'xtick', [], 'ytick', [])
        caxis([-caxtmp caxtmp]); box on;
        if ind+20*(floor(layers/10)) == 1; ylabel('Conv1'); title('Fig 2a.'); end
        if ind+20*(floor(layers/10)) == 21; ylabel('Conv5'); end
        if ind+20*(floor(layers/10)) == 2; title('Pre-trained AlexNet'); end
    end
    tmp = net_rand.Layers(layers).Weights;sztmp = 1:size(tmp,4);indtmpp=datasample(sztmp, vis_N,'Replace', false);
    for ind = 1:length(indtmpp)
        tmp2 = squeeze(tmp(:,:,1,indtmpp(ind)));caxtmp = max(abs(min(tmp2(:))), abs(max(tmp2(:))));
        ax=subplot(3,10,ind+4+20*(floor(layers/10)));imagesc(squeeze(tmp(:,:,1,indtmpp(ind))));axis image off;caxis(ax, [-caxtmp caxtmp])
        if ind+4+20*(floor(layers/10)) == 6; title('Permuted AlexNet'); end
    end
    colormap(gray);
end

subplot(3,10,[9 10 19 20 29 30])
load('IMAGENET_ALEXNET_for_test.mat')
top1_accu_pret = squeeze(accuracy_tot(:, 1, 1)); top1_accu_perm = squeeze(accuracy_tot(:, 2, 1));
b1 = bar([1], [mean(top1_accu_pret)], 'FaceColor', [0.5 0.5 0.5]); ylim([0 0.8])
x_min = std(top1_accu_pret); x_max = std(top1_accu_pret);
hold on; errorbar(1, mean(top1_accu_pret), x_min, x_max, 'k')
b2 = bar(2, mean(top1_accu_perm), 'r');
x_min = std(top1_accu_perm); x_max = std(top1_accu_perm);
hold on; errorbar(2, mean(top1_accu_perm), x_min, x_max, 'k')
errorbarlogy; ylim([0.0001 1]); hold on; plot([0, 3], [0.001 0.001])
ptmp = ranksum(top1_accu_pret, top1_accu_perm, 'tail', 'right');
title('Fig 2b. Image classification performance');
set(gca, 'xtick', [1 2], 'xticklabel', {'Pre-trained', 'Permuted'});
xlabel('Type of AlexNet')
ylabel('Performance')
set(gcf, 'units','normalized','outerposition',[0 0.5 0.8 0.5])

figure;
subplot(4,6,[1 2 7 8])
mattmp = zeros(13,13);
indtmps = indseg(PN_ex);
[ic,jc,~] = ind2sub([13, 13, 256], indtmps);
for ii = 1:length(PN_ex)
    mattmp(ic(ii),jc(ii)) = 1;
    colormap(gray)
end
imagesc(mattmp);
for ii = 1:length(PN_ex)
    hold on
    text(jc(ii)-0.1, ic(ii),['#' num2str(ii)])
end
for ii = 1:12
    hold on
    plot([0.5+ii, 0.5+ii], [0.5 13.5], 'r')
    plot([0.5 13.5], [0.5+ii, 0.5+ii], 'r')
end
axis image; title('Activation map in Conv5')

% sgtitle('Fig. 2c: Number neurons in permuted AlexNet')
for ii = 1:length(PN_ex)
    if ii<3
        subplot(4,6,ii+2)
    else
        subplot(4,6,ii+6)
    end
    hold on
    shadedErrorBar(number_sets, resp_totmean_tmp(:,PN_ex(ii)), resp_totstd_tmp(:,PN_ex(ii)))
    if ismember(ii, [3,4]); xlabel('Numerosity'); end
    if ismember(ii, [1,3]); ylabel('Response'); end    
    plot(number_sets, resp_standard_tmp(:,PN_ex(ii)), 'b' )
    plot(number_sets, resp_ctr1_tmp(:,PN_ex(ii)), 'r' )
    plot(number_sets, resp_ctr2_tmp(:,PN_ex(ii)), 'g')
    % legend({'tot','std','ctr1', 'ctr2'});% scatter(number_sets, resp_mean(:,randi(43264)))
        set(gca, 'xtick', 0:10:30)
    title(['Neuron #' num2str(ii)])
end

subplot(2,3,3); hold on;
b1 = plot(number_sets, PN_monkey); alpha (0.3); hold on;
b2 = shadedErrorBar(number_sets, mean(PNdist_perm_tot,1), std(PNdist_perm_tot,[],1)); alpha (0.3);%b2.EdgeColor = 'none'; %/sum(PN_popul_RP));
% b3 = plot(number_sets, PN_pretrained); alpha (0.3)
set(gca, 'xtick', 0:10:30); xlim([0 30])
xlabel('Preferred numerosity'); ylabel('Ratio')
title('Fig 2d. Distribution of preferred numerosity')
legend([b1, b2.mainLine], {'monkey', 'permuted'})

subplot(2,3,4); hold on;
xtmp = number_sets;sigmas = zeros(1,length(number_sets)); R2 = zeros(1,length(number_sets)); TNtmp = tuning_curve_ave_per;
for ii = 1:length(number_sets)
    hh=plot(number_sets, TNtmp(ii,:), 'Color', colortmp(ii,:));
    ytmp = TNtmp(ii,:);
    if isnan(ytmp)
        sigmas(ii) = nan;
    else
        %         f = fit(xtmp.', ytmp.', 'gauss1', options);
        %         sigmas(ii) = f.c1/2; xfinetmp = 1:0.01:30;
        %         tmp = f.a1*exp(-((xfinetmp-f.b1)/f.c1).^2);
        %         ymean = nanmean(ytmp);
        %     SStot = sum((ytmp-ymean).^2); %     SSres = sum((ytmp-tmp).^2); %     R2(ii) = 1-SSres./SStot;
    end
    %     hold on; h=plot(xfinetmp, tmp, 'Color', colortmp(ii,:));legend off;
    %     yysig = 0:0.5/16:0.5;
    %     plot([f.b1, f.b1+sigmas(ii)], [yysig(ii+1), yysig(ii+1)], 'Color', colortmp(ii,:))
end
set(gca, 'xtick', 0:10:30, 'ytick', [0 1])
xlabel('Preferred numerosity'); ylabel('Normalized response'); title('Fig 2e. Average tuning curve')

subplot(2,3,5); hold on;
sig_lin_per = mean(sig_lin_per_tot, 1);
sig_log_per = mean(sig_log_per_tot, 1);
stdlin = std(sig_lin_per_tot, [],1);
stdlog = std(sig_log_per_tot, [],1);
s1 = scatter(number_sets, sig_lin_per, 'MarkerEdgeColor', color_linear, 'MarkerFaceColor', color_linear);
p1 = polyfit(number_sets, sig_lin_per, 1);
%errorbar(number_sets, sig_lin_per, stdlin, 'LineStyle', 'none');
plot(number_sets, p1(1)*number_sets+p1(2), 'color', color_linear)
s2 = scatter(number_sets, sig_log_per, 'MarkerEdgeColor', color_log, 'MarkerFaceColor', color_log);
%errorbar(number_sets, sig_log_per, stdlog, 'LineStyle', 'none')
p2 = polyfit(number_sets, sig_log_per, 1);
plot(number_sets, p2(1)*number_sets+p2(2), 'color', color_log)
xlabel('Preferred numerosity');
ylabel('\sigma of Gaussian fit')
[r1,pv1] = corrcoef(number_sets, sig_lin_per);
[r2,pv2] = corrcoef(number_sets, sig_log_per);

s3 = scatter(x_axis, sig_monkey_linear, 'MarkerEdgeColor', color_linear, 'MarkerFaceColor', 'none');
s4 = scatter(x_axis, sig_monkey_log, 'MarkerEdgeColor', color_log, 'MarkerFaceColor', 'none');

scatter(-0.5, 15, 20, 'o', 'MarkerEdgeColor', color_linear, 'MarkerFaceColor', color_linear)
text(1, 15, 'Linear', 'fontsize', 8)
scatter(-0.5, 13.3, 20, 'o', 'MarkerEdgeColor', color_log, 'MarkerFaceColor', color_log)
text(1, 13.3, 'Log', 'fontsize', 8)
scatter(-0.5, 11.7, 20, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k')
text(1, 11.7, 'Permuted', 'fontsize', 8)
scatter(-0.5, 10.1, 20, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'none')
text(1, 10.1, 'Monkey', 'fontsize', 8)
set(gca, 'xtick', 0:10:30, 'ytick', 0:8:16);
xlim([-2 32]); ylim([-2 16])
title('Fig 2f. Weber-Fechner law')

subplot(2,3,6); hold on;
tmp1 = (performance_pret_perm_sets(:,1)); % performance of pre-trained network
tmp2 = (performance_pret_perm_sets(:,2)); % performance of permuted network
tmp3 = (performance_pret_perm_sets(:,3)); % shuffled response
tmp4 = (performance_pret_perm_sets(:,4)); % shuffled response
b1=bar([1], [mean(tmp1)], 'k'); hold on
b2=bar([2], [mean(tmp2)], 'r');
b3=bar([3], [mean(tmp4)], 'w');
errorbar([1,2,3], [mean(tmp1), mean(tmp2), mean(tmp4)], [std(tmp1), std(tmp2), std(tmp4)], 'b','LineStyle', 'none');
x_lim = xlim;
plot(x_lim, [0.5, 0.5], 'k--')
set(gca, 'xtick', [], 'ytick', 0.4:0.1:0.8);
ylabel('Performance')
legend([b1, b2, b3], {'Pre-trained','Permuted','Response shuffled'})
ptmp1 = ranksum(tmp1, tmp3, 'tail', 'right');
ptmp3 = ranksum(tmp1, tmp2, 'tail', 'left');
ylim([0.4 0.8]);
title('Fig. 2h: Task performance');

set(gcf, 'units','normalized','outerposition',[0.2 0.05 0.8 0.7])
