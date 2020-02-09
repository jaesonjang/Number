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

% This code performs a demo simulation for Fig. 1 in the manuscript.
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
savedir = [pathtmp '\Dataset\Data\Fig1_generated_data'];

%% Setting parameters

rand_layers_ind = [2, 6, 10, 12 14];    % Index of convolutional layer of AlexNet
number_sets = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]; % Candidiate numerosities of stimulus
LOI = 'relu5';  % Name of layer at which the activation will be measured
image_iter = 10;  % Number of images for a given condition
p_th1 = 0.01; p_th2 = 0.01; p_th3 = 0.01;  % Significance levels for two-way ANOVA
vis_N = 4;      % Number of example filters
PN_ex = [1, 5, 12, 16];  % Example PN
% p_thtmp = 0.1; % Visualizing p value threshold

issavefig = 1;   % save fig files?

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

%% Figure 1d. Stimulus set
ff = figure; set(gcf,'Visible', 'off')

%% standard
indtmp = randi(size(image_sets_standard,3));
subplot(3,4,1);tmp = squeeze(image_sets_standard(:,:, indtmp,3));imagesc(tmp);colormap(gray);axis image xy off; ylabel('Set 1')
subplot(3,4,2);tmp = squeeze(image_sets_standard(:,:, indtmp,9));imagesc(tmp);colormap(gray);axis image xy off
title('Fig. 1d. Examples of stimuli used to measure number tuning')
subplot(3,4,3);tmp = squeeze(image_sets_standard(:,:, indtmp,15));imagesc(tmp);colormap(gray);axis image xy off

%% control 1
subplot(3,4,5);tmp = squeeze(image_sets_control1(:,:, indtmp,3));imagesc(tmp);colormap(gray);axis image xy off; ylabel('Set 2')
subplot(3,4,6);tmp = squeeze(image_sets_control1(:,:, indtmp,9));imagesc(tmp);colormap(gray);axis image xy off
subplot(3,4,7);tmp = squeeze(image_sets_control1(:,:, indtmp,15));imagesc(tmp);colormap(gray);axis image xy off

%% control 2
polyxx = squeeze(polyxy(:,:,:,1)); polyyy = squeeze(polyxy(:,:,:,2));
subplot(3,4,9);tmp = squeeze(image_sets_control2(:,:, indtmp,3));imagesc(tmp);colormap(gray);axis image xy off; xlabel('N = 4');ylabel('Set 3')
pgon = polyshape(squeeze(polyxx(3,indtmp, :,:)),squeeze(polyyy(3,indtmp, :,:)));hold on;plot(pgon)
subplot(3,4,10);tmp = squeeze(image_sets_control2(:,:, indtmp,9));imagesc(tmp);colormap(gray);axis image xy off; xlabel('N = 16');
pgon = polyshape(squeeze(polyxx(9,indtmp, :,:)),squeeze(polyyy(9,indtmp, :,:)));hold on;plot(pgon)
subplot(3,4,11);tmp = squeeze(image_sets_control2(:,:, indtmp,15));imagesc(tmp);colormap(gray);axis image xy off; xlabel('N = 28');
pgon = polyshape(squeeze(polyxx(15,indtmp, :,:)),squeeze(polyyy(15,indtmp, :,:)));hold on;plot(pgon)

for ii = 1:length(number_sets)
    numtmp = number_sets(ii); circle_radius = 7;
    areaSum = numtmp*(pi*circle_radius.^2); scalingtmp = sqrt(areaSum/1200);
    circle_radiuss(ii) = circle_radius/scalingtmp;
end
subplot(3,4,8);hold on;
yyaxis left; plot(number_sets, circle_radiuss);ylabel('Area of each dot')
yyaxis right; plot(number_sets, 1200+zeros(1,length(number_sets)))
yticks([1200]);ylabel('Total area of dots');xlabel('Numerosity')
title('Area profiles for Set 2')
if issavefig; savefig([pathtmp '\Figs\1d']); end
close(ff)

%% Finding number selective neurons
if generatedat    
    response_NS_sets = cell(iter, 2);
    ind_NS_sets = cell(iter, 2); %% 2D: dim1 : iteration, dim2: pretrained/permuted
    
    for iterind = 1:iter        
        %% Calculating response to stimulus
        disp(['[Step 2/6] Calculating response to stimuli...'])
        response_tot_standard = getactivation(net, LOI, image_sets_standard);
        response_tot_control1 = getactivation(net, LOI, image_sets_control1);
        response_tot_control2 = getactivation(net, LOI, image_sets_control2);
        % Getting total response matrix
        response_tot = cat(2,response_tot_standard, response_tot_control1, response_tot_control2);
        units_N = size(response_tot, 3);
        % Note: response_tot is three dimensional matrix (16 X 150 X 43264)
        % where dim1 : number of tested numerosities, dim2: number of images,
        % dim3: number of neuron of relu5
        
        %% Getting p-values of two-way ANOVA from response
        disp(['[Step 3/6] Obtaining p-values for two-way ANOVA test...'])
        pvalues = getpv(response_tot); % using two-way ANOVA
        % pvalues2 = getpvforeach(response_tot); % one-way ANOVA for each set, for visualizing clear samples
        
        %% Analyzing p-values to find number selective neurons
        disp(['[Step 4/6] Analyzing p-values to find number selective neurons...'])
        pv1 = pvalues(1,:); pv2 = pvalues(2,:);pv3 = pvalues(3,:);
        ind1 = (pv1<p_th1);ind2 = (pv2>p_th2);ind3 = (pv3>p_th2);
        ind_NS = find(ind1.*ind2.*ind3); % indices of number selective units
        
        %% Calculating mean response
        disp(['[Step 5/6] Calculating average response...'])
        resp_mean = squeeze((mean(response_tot, 2))); resp_std = squeeze(std(response_tot, 0,2));
        resp_mean_standard = squeeze((mean(response_tot_standard, 2))); resp_std_standard = squeeze(std(response_tot_standard, 0,2));
        resp_mean_control1 = squeeze((mean(response_tot_control1, 2))); resp_std_control1 = squeeze(std(response_tot_control1, 0,2));
        resp_mean_control2 = squeeze((mean(response_tot_control2, 2))); resp_std_control2 = squeeze(std(response_tot_control2, 0,2));
        
        %% Calculating preferred number of number neurons
        disp(['[Step 6/6] Calculating the preferred numerosity...'])
        response_NS_tot = response_tot(:,:,ind_NS);
        response_NS_mean = squeeze(mean(response_NS_tot, 2));
        [M,PNind] = max(response_NS_mean); %% PNind : preferred number of number selective neurons
        units_PN = zeros(1,units_N)/0; units_PN(ind_NS) = PNind; % preferred number for each neuron
        tmp1 = response_NS_tot(:,1:image_iter, :); tmp1 = (mean(tmp1, 2));
        tmp2 = response_NS_tot(:,image_iter+1:image_iter*2, :);tmp2 = (mean(tmp2, 2));
        tmp3 = response_NS_tot(:,image_iter*2+1:image_iter*3,:);tmp3 = (mean(tmp3, 2));
        response_NS_sep = cat(2,tmp1, tmp2, tmp3);

        %% Getting tuning width and goodness of fit (Gaussian) in log scale
        % xtmp = log2(number_sets);
        % [sigmas, R2s] = getlogfit_individual(ind_NS, response_NS_mean, xtmp);
        
        %% Getting example tuning curves for individual neurons
        % pv4 = pvalues2(2,:); pv5 = pvalues2(1,:); pv6 = pvalues2(3,:);
        % ind4 = pv4<p_thtmp; ind5 = pv5<p_thtmp; ind6 = pv6<p_thtmp;
        % isNS2 = (ind1.*ind2.*ind3.*ind4.*ind5.*ind6);
        resp_totmean_tmp = zeros(length(number_sets), length(PN_ex)); resp_totstd_tmp = zeros(length(number_sets), length(PN_ex));
        resp_standard_tmp = zeros(length(number_sets), length(PN_ex)); resp_ctr1_tmp = zeros(length(number_sets), length(PN_ex));
        resp_ctr2_tmp = zeros(length(number_sets), length(PN_ex)); indseg = zeros(1,length(number_sets));
        for ii = 1:length(number_sets)
            PNtmp = ii;
            % indcand = find(units_PN ==PNtmp & isNS2);
            indcand = find(units_PN ==PNtmp);
            if length(indcand)>0
                indcand2 = datasample(indcand, 1);
                resp_totmean_tmp(:,ii) = resp_mean(:, indcand2);
                resp_totstd_tmp(:,ii) = resp_std(:, indcand2)/sqrt(3*image_iter);
                resp_standard_tmp(:,ii) = resp_mean_standard(:, indcand2);
                resp_ctr1_tmp(:,ii) = resp_mean_control1(:, indcand2);
                resp_ctr2_tmp(:,ii) = resp_mean_control2(:, indcand2);
                indseg(ii) = indcand2;
            end
            
        end
        if savedat
            save([savedir '\1e_Exampleresponses'], 'resp_totmean_tmp', ...
                'resp_totstd_tmp', 'resp_standard_tmp', 'resp_ctr1_tmp', 'resp_ctr2_tmp', 'indseg' );
        end
    
    response_NS_sets{iterind, 1} = response_NS_sep;
    ind_NS_sets{iterind,1} = units_PN;
    
    end
else
    load([pathtmp '\Dataset\Data\1e_Exampleresponses']);
    % load data for pretrained and permuted network, 100 iterations
    load('Data_NSsimulationfor100iterations')
end

%% Figure 1e. Number neurons in pre-trained AlexNet
    ff = figure; set(gcf,'Visible', 'off')
    sgtitle('Fig. 1e: Number neurons in pre-trained AlexNet')
    for ii = 1:length(PN_ex)
        subplot(2,2,ii)
        hold on
        shadedErrorBar(number_sets, resp_totmean_tmp(:,PN_ex(ii)), resp_totstd_tmp(:,PN_ex(ii)))
        xlabel('PN');ylabel('Response')
        plot(number_sets, resp_standard_tmp(:,PN_ex(ii)), 'b')
        plot(number_sets, resp_ctr1_tmp(:,PN_ex(ii)), 'r' )
        plot(number_sets, resp_ctr2_tmp(:,PN_ex(ii)), 'g')
    end
    if issavefig; savefig([pathtmp '\Figs\1e']); end
    close(ff)

%% Analyzing data
if generatedat
    [tuning_curve_ave, sig_lin_pre_tot, R2_lin_pre_tot, ...
        sig_log_pre_tot, R2_log_pre_tot ]...
        = analyzefig1data(iter, number_sets, response_NS_sets, ind_NS_sets);
    if savedat
        save([savedir '\1fg_averagetuningcurves'], 'tuning_curve_ave', 'sig_lin_pre_tot', 'R2_lin_pre_tot', ...
            'sig_log_pre_tot', 'R2_log_pre_tot');
    end
else
    load([pathtmp '\Dataset\Data\1fg_averagetuningcurves'])
end

%% figure 1f,g
col = [0 0/4470 0.7410;0.85 0.325 0.098;0.929 0.694 0.125;...
    0.494 0.184 0.556;0.466 0.674 0.188;0.301 0.745 0.933;0.635 0.0780 0.1840];
colortmp = cat(1, col, col, col, col);

ff = figure; set(gcf,'Visible', 'off');hold on; options = fitoptions('gauss1', 'Lower', [0 0 0], 'Upper', [1.5 30 100]);
xtmp = number_sets; sigmas = zeros(1,length(number_sets)); R2 = zeros(1,length(number_sets)); TNtmp = tuning_curve_ave;
for ii = 1:length(number_sets)
    hh=plot(number_sets, tuning_curve_ave(ii,:), 'Color', colortmp(ii,:));
    ytmp = TNtmp(ii,:);
    if isnan(ytmp) ; sigmas(ii) = nan;
    else
        f = fit(xtmp.', ytmp.', 'gauss1', options); sigmas(ii) = f.c1/2; xfinetmp = 1:0.01:30;
        tmp = f.a1*exp(-((xfinetmp-f.b1)/f.c1).^2); ymean = nanmean(ytmp);
    end
    %     hold on; h=plot(xfinetmp, tmp, 'Color', colortmp(ii,:));legend off; yysig = 0:0.5/16:0.5;
    %     plot([f.b1, f.b1+sigmas(ii)], [yysig(ii+1), yysig(ii+1)], 'Color', colortmp(ii,:))
end
xlabel('Numerosity'); ylabel('Normalized response'); title('Fig 1f: Average tuning curve')
if issavefig; savefig([pathtmp '\Figs\1f']); end
close(ff)

%% Loading monkey data from Nieder et al., 2007
%save('sig_data_Weber_nieder2007', 'sig_linear', 'sig_log', 'x_axis')
load('sigma_from_monkey_Nieder_2007.mat')       % Loding monkey data from Nieder et al., 2007
ff = figure; set(gcf,'Visible', 'off'); hold on
sig_lin_pre = mean(sig_lin_pre_tot, 1); sig_log_pre = mean(sig_log_pre_tot, 1);
scatter(number_sets, sig_lin_pre, 'k', 'fill')
p1 = polyfit(number_sets, sig_lin_pre, 1); plot(number_sets, p1(1)*number_sets+p1(2), 'k');
scatter(number_sets, sig_log_pre, 'r', 'fill')
p2 = polyfit(number_sets, sig_log_pre, 1);
plot(number_sets, p2(1)*number_sets+p2(2), 'r')
xlabel('Numerosity'); ylabel('Sigma of Gaussian fit'); ylim([0 16]); title('Fig 1g: Weber-Fechner law observed')
[r1,pv1] = corrcoef(number_sets, sig_lin_pre); [r2,pv2] = corrcoef(number_sets, sig_log_pre);
hold on; scatter(x_axis, sig_monkey_linear, 'k'); scatter(x_axis, sig_monkey_log, 'r')
legend('linear', ['r = ' num2str(r1(2)) ', p = ' num2str(pv1(2)) ', line eq: y = ' num2str(p1(1)) 'x + ' num2str(p1(2))],...
    'log2', ['r = ' num2str(r2(2)) ', p = ' num2str(pv2(2)) ', line eq: y = ' num2str(p2(1)) 'x + ' num2str(p2(2))], 'lin data', 'log data')
if issavefig; savefig([pathtmp '\Figs\1g']); end
close(ff)

%% Visualizing figures
fig1d = openfig([pathtmp '\Figs\1d'], 'invisible');
set(gcf, 'units','normalized','outerposition',[0 0.4 0.5 0.6], 'visible', 'on')

figure;
subplot(2,5,[1 2 6 7])
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

sgtitle('Number neurons in the pre-trained AlexNet')
for ii = 1:length(PN_ex)
    if ii<=2; subplot(2,5,ii+2); else; subplot(2,5,ii+5); end
    hold on
    shadedErrorBar(number_sets, resp_totmean_tmp(:,PN_ex(ii)), resp_totstd_tmp(:,PN_ex(ii)))
    if ismember(ii, [3,4]); xlabel('Numerosity'); end
    if ismember(ii, [1,3]); ylabel('Response'); end    
    plot(number_sets, resp_standard_tmp(:,PN_ex(ii)), 'b')
    plot(number_sets, resp_ctr1_tmp(:,PN_ex(ii)), 'r' )
    plot(number_sets, resp_ctr2_tmp(:,PN_ex(ii)), 'g')
    set(gca, 'xtick', 0:10:30)
    title(['Number neuron #' num2str(ii)])
end

subplot(2,5,5); hold on; options = fitoptions('gauss1', 'Lower', [0 0 0], 'Upper', [1.5 30 100]);
xtmp = number_sets; sigmas = zeros(1,length(number_sets)); R2 = zeros(1,length(number_sets)); TNtmp = tuning_curve_ave;
for ii = 1:length(number_sets)
    hh=plot(number_sets, tuning_curve_ave(ii,:), 'Color', colortmp(ii,:));
end
set(gca, 'xtick', 0:10:30, 'ytick', [0 1])
xlabel('Preferred numerosity'); ylabel('Normalized response'); title('Fig 1f. Average tuning curve')

subplot(2,5,10); hold on
sig_lin_pre = mean(sig_lin_pre_tot, 1); sig_log_pre = mean(sig_log_pre_tot, 1);
stdlin = std(sig_lin_pre_tot, [],1); stdlog = std(sig_log_pre_tot, [],1);
scatter(number_sets, sig_lin_pre, 'MarkerEdgeColor', color_linear, 'MarkerFaceColor', color_linear)
p1 = polyfit(number_sets, sig_lin_pre, 1);
plot(number_sets, p1(1)*number_sets+p1(2), 'color', color_linear);
scatter(number_sets, sig_log_pre, 'MarkerEdgeColor', color_log, 'MarkerFaceColor', color_log)
p2 = polyfit(number_sets, sig_log_pre, 1);
plot(number_sets, p2(1)*number_sets+p2(2), 'color', color_log)
xlabel('Preferred numerosity');
ylabel('\sigma of Gaussian fit')
title('Fig 1g: Weber-Fechner law observed')
[r1,pv1] = corrcoef(number_sets, sig_lin_pre); [r2,pv2] = corrcoef(number_sets, sig_log_pre);
scatter(x_axis, sig_monkey_linear, 'MarkerEdgeColor', color_linear, 'MarkerFaceColor', 'none');
scatter(x_axis, sig_monkey_log, 'MarkerEdgeColor', color_log, 'MarkerFaceColor', 'none');
set(gca, 'xtick', 0:10:30, 'ytick', 0:8:16)
xlim([-2 32]); ylim([-2 16])

scatter(-0.5, 15, 20, 'o', 'MarkerEdgeColor', color_linear, 'MarkerFaceColor', color_linear)
text(1, 15, 'Linear', 'fontsize', 8)
scatter(-0.5, 13.3, 20, 'o', 'MarkerEdgeColor', color_log, 'MarkerFaceColor', color_log)
text(1, 13.3, 'Log', 'fontsize', 8)
scatter(-0.5, 11.7, 20, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k')
text(1, 11.7, 'Pre-trained', 'fontsize', 8)
scatter(-0.5, 10.1, 20, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'none')
text(1, 10.1, 'Monkey', 'fontsize', 8)
set(gcf, 'units','normalized','outerposition',[0.2 0.1 0.8 0.6])
