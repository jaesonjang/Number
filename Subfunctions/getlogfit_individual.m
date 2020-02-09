function [sigma, R2] = getlogfit_individual(ind_NS, response_NS_mean, xtmp)

sigma = zeros(1,length(ind_NS));
R2 = zeros(1,length(ind_NS));


for ii = 1:length(ind_NS)
    
ytmp = response_NS_mean(:,ii)';
options = fitoptions('gauss1', 'Lower', [0 0 0], 'Upper', [1.5*max(ytmp) 30 100]);
if isnan(ytmp)
    sigma(ii) = nan;
else
    f = fit(xtmp.', ytmp.', 'gauss1', options);
    sigma(ii) = f.c1/sqrt(2); 
    tmp = f.a1*exp(-((xtmp-f.b1)/f.c1).^2);
    ymean = nanmean(ytmp);
    SStot = sum((ytmp-ymean).^2);SSres = sum((ytmp-tmp).^2);
    R2(ii) = 1-SSres./SStot;
end
end

end