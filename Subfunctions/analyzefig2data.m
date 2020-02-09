
function [tuning_curve_ave, sig_lin_per_tot, R2_lin_per_tot, ...
    sig_log_per_tot, R2_log_per_tot ]...
    = analyzefig2data(iter, number_sets, response_NS_sets, ind_NS_sets)

% ind_NS_sets: PN of each neuron
% response_NS_sets: average response of each neuron for each stimulus set 

tuning_curve_sum = zeros(length(number_sets),length(number_sets));
sig_lin_per_tot = zeros(iter, 16);
R2_lin_per_tot = zeros(iter, 16);
sig_log_per_tot = zeros(iter, 16);
R2_log_per_tot = zeros(iter, 16);
for ii = 1:iter
        %% get tuning curve data
        response_NS_sep = response_NS_sets{ii, 2};
        response_NS_mean = squeeze(mean(response_NS_sep,2));
        tmp = ind_NS_sets{ii,2};
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
        [sig_lin_per, R2_lin_per] = Weberlaw(TNCV_tmp, xtmp);
        sig_lin_per_tot(ii,:) = sig_lin_per;
        R2_lin_per_tot(ii,:) = R2_lin_per;
        xtmp = log2(number_sets);
        [sig_log_per, R2_log_per] = Weberlaw(TNCV_tmp, xtmp);
        sig_log_per_tot(ii,:) = sig_log_per;
        R2_log_per_tot(ii,:) = R2_log_per;
        
    end
    tuning_curve_ave = tuning_curve_sum/iter;
    
end