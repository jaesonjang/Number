function indNS_L1_str = getPNconsistentind(response_NS_tot_L1)
%% input : response of NS units
response_mattmp = response_NS_tot_L1;

PNs_L1 = zeros(size(response_NS_tot_L1,2), size(response_NS_tot_L1,3));
for ii = 1:size(response_NS_tot_L1,3)
    for jj = 1:size(response_NS_tot_L1,2)
        TCtmp = squeeze(response_mattmp(:,jj,ii));
        tmp = find(TCtmp == max(TCtmp));
        if length(tmp) ~= 1
             PNs_L1(jj,ii) = nan;
           %   PNs_L1(jj,ii) = randi(16);
        else
        PNs_L1(jj,ii) = find(TCtmp == max(TCtmp));
        end
    end
end
PNs_L1_std = PNs_L1(1:50, :);
PNs_L1_ctr1 = PNs_L1(51:100, :);
PNs_L1_ctr2 = PNs_L1(101:150, :);

PN_diversity_L1 = nanstd(PNs_L1, 1);
PN_diversity_L1_std = nanstd(PNs_L1_std, 1);
PN_diversity_L1_ctr1 = nanstd(PNs_L1_ctr1, 1);
PN_diversity_L1_ctr2 = nanstd(PNs_L1_ctr2, 1);
figure
subplot(1,3,1);histogram(PN_diversity_L1_std)
subplot(1,3,2);histogram(PN_diversity_L1_ctr1)
subplot(1,3,3);histogram(PN_diversity_L1_ctr2)
th = 3;th2=1;
indNS_L1_str = (find((PN_diversity_L1_std<th).*(PN_diversity_L1_ctr1<th).*(PN_diversity_L1_ctr2<th)));
% indNS_L5_str = (find((PN_diversity_L5_std<th).*(PN_diversity_L5_ctr1<th).*(PN_diversity_L5_ctr2<th)));
% indNS_L1_CV = (Units_CV_L1<1);
% indNS_L5_CV = (Units_CV_L5<1);

% indtmpL1 = (find((PN_diversity_L1_std<th).*(PN_diversity_L1_ctr1<th).*(PN_diversity_L1_ctr2<th).*(Units_CV_L1<2)));
end
% indtmpL5 = (find((PN_diversity_L5_std<th).*(PN_diversity_L5_ctr1<th).*(PN_diversity_L5_ctr2<th).*(Units_CV_L5<2)))
% Units_PN_L1
% 
% %% test1
% for iter = 1:16
% indtmp = randi(length(indtmpL1));
% indtmpp = indtmpL1(indtmp);
% figure (1)
% subplot(4,4,iter)
% hold on
% 
% tmp = squeeze(response_NS_tot_L1(:,:, indtmpp));
% tmp2 = mean(tmp, 2);
% for ii = 1:size(response_NS_tot_L1,2)
%     plot(squeeze(response_NS_tot_L1(:,ii, indtmpp)))
%    
% end
% figure (2)
% subplot(4,4,iter)
% hold on
% plot(tmp2, 'k')
% end
% for iter = 1:16
% indtmp = randi(length(indtmpL5));
% indtmpp = indtmpL5(indtmp);
% figure (3)
% subplot(4,4,iter)
% hold on
% 
% tmp = squeeze(response_NS_tot_L5(:,:, indtmpp));
% tmp2 = mean(tmp, 2);
% for ii = 1:size(response_NS_tot_L5,2)
%     plot(squeeze(response_NS_tot_L5(:,ii, indtmpp)))
%    
% end
% figure (4)
% subplot(4,4,iter)
% hold on
% plot(tmp2, 'k')
% end
% 
% % PN_nan_L1 = zeros(1,size(PNs_L1,2));
% % for ii = 1:size(PNs_L1,2)
% % PN_nan_L1(ii) = length(find(isnan(PNs_L1(:,ii))));
% % end
% % figure
% % histogram(PN_nan_L1)
% % 
% % PN_nan_L5 = zeros(1,size(PNs_L5,2));
% % for ii = 1:size(PNs_L5,2)
% % PN_nan_L5(ii) = length(find(isnan(PNs_L5(:,ii))));
% % end
% % histogram(PN_nan_L5)
% % 
% % histogram(PN_diversity_L1)
% % histogram(PNs_L1)
% 
% 
% response_mattmp = response_NS_tot_L5;
% PNs_L5 = zeros(size(response_NS_tot_L5,2), size(response_NS_tot_L5,3));
% for ii = 1:size(response_NS_tot_L5,3)
%     for jj = 1:size(response_NS_tot_L5,2)
%         TCtmp = squeeze(response_mattmp(:,jj,ii));
%         tmp = find(TCtmp == max(TCtmp));
%         if length(tmp) ~= 1
%              PNs_L5(jj,ii) = nan;
%              %PNs_L5(jj,ii) = randi(16);
%         else
%         PNs_L5(jj,ii) = find(TCtmp == max(TCtmp));
%         end
%     end
% end
% PNs_L5_std = PNs_L5(1:50, :);
% PNs_L5_ctr1 = PNs_L5(51:100, :);
% PNs_L5_ctr2 = PNs_L5(101:150, :);
% 
% PN_diversity_L5 = nanstd(PNs_L5, 1);
% PN_diversity_L5_std = nanstd(PNs_L5_std, 1);
% PN_diversity_L5_ctr1 = nanstd(PNs_L5_ctr1, 1);
% PN_diversity_L5_ctr2 = nanstd(PNs_L5_ctr2, 1);
% figure
% subplot(1,3,1);histogram(PN_diversity_L5_std)
% subplot(1,3,2);histogram(PN_diversity_L5_ctr1)
% subplot(1,3,3);histogram(PN_diversity_L5_ctr2)
% 
% % 
% % for ii = 1:size(response_NS_tot_L1,3)
% %     for jj = 1:size(response_NS_tot_L1,2)
% %         TCtmp = squeeze(response_mattmp(:,jj,ii))
% %     end
% % end