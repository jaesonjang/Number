
function [tuning_curve_ave, sig_lin_pre_tot, R2_lin_pre_tot, ...
    sig_log_pre_tot, R2_log_pre_tot ]...
    = analyzefig1data(iter, number_sets, response_NS_sets, ind_NS_sets)

% ind_NS_sets: PN of each neuron
% response_NS_sets: average response of each neuron for each stimulus set 
% sigmas_sets: tuning width and R2 of each neuron

tuning_curve_sum = zeros(length(number_sets),length(number_sets));
sig_lin_pre_tot = zeros(iter, 16);
R2_lin_pre_tot = zeros(iter, 16);
sig_log_pre_tot = zeros(iter, 16);
R2_log_pre_tot = zeros(iter, 16);
for ii = 1:iter
    %% get tuning curve data
    response_NS_sep = response_NS_sets{ii, 1};
    response_NS_mean = squeeze(mean(response_NS_sep,2));
    tmp = ind_NS_sets{ii,1};
    PNind = tmp(tmp>0);
    
    tuning_curve = zeros(length(number_sets),length(number_sets));
    for jj = 1:length(number_sets)
        tuning_curvetmp = response_NS_mean(:,find(PNind==jj));
        cvtmp = mean(tuning_curvetmp, 2);
        tuning_curve(jj,:) = rescale(cvtmp);
    end
    tuning_curve_sum = tuning_curve_sum+tuning_curve;
        
    %% get Weber's law data
    TNCV_tmp = tuning_curve;
    xtmp = number_sets;
    [sig_lin_pre, R2_lin_pre] = Weberlaw(TNCV_tmp, xtmp);
    sig_lin_pre_tot(ii,:) = sig_lin_pre;
    R2_lin_pre_tot(ii,:) = R2_lin_pre;
    xtmp = log2(number_sets);
    [sig_log_pre, R2_log_pre] = Weberlaw(TNCV_tmp, xtmp);
    sig_log_pre_tot(ii,:) = sig_log_pre;
    R2_log_pre_tot(ii,:) = R2_log_pre;  
end
tuning_curve_ave = tuning_curve_sum/iter;
end