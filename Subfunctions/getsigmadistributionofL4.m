function [sigmastotdec, sigmastotinc, sigmastotdeclog, sigmastotinclog] ...
    = getsigmadistributionofL4

iter = 100;
sigmastotdec = [];
sigmastotinc = []; 
sigmastotdeclog = [];
sigmastotinclog = []; 
R2stotdec = [];
R2stotinc = [];
for iterind = 1:iter
    tic
    lind = 4;
    load(['D:\MATLAB\Number selectivity\191211_data_sigmaL4linlog\data191208_iter_' num2str(iterind)])
    resptmp = mean(response_mean_layers{lind},2);
    NS_PN = NS_PN_layers{4};
    NS_PN = NS_PN(NS_PN>0);
    
    indtmp1 = (NS_PN==1);
    resptmp1 = resptmp(:,indtmp1);
    respmeantmp1 = mean(resptmp1);
    [sorttmp,I]= sort(respmeantmp1);
    respthind1 = I(ceil(length(I)/10):end);
    

    indtmp2 = (NS_PN==16);
    resptmp2 = resptmp(:,indtmp2);
    respmeantmp2 = mean(resptmp2);
    [sorttmp,I]= sort(respmeantmp2);
    respthind2 = I(ceil(length(I)/10):end);
    
    
    
    sigmastmp = sigmas_layers{lind, 1};
    sigmastmp = sigmastmp(respthind1);
    sigmastotdec = [sigmastotdec, sigmastmp];
    R2stmp = R2s_layers{lind, 1};
    R2stmp = R2stmp(respthind1);
    R2stotdec = [R2stotdec, R2stmp];
    
    sigmastmp = sigmas_layers{lind, 2};
    sigmastmp = sigmastmp(respthind2);
    R2stmp = R2s_layers{lind, 2};
    R2stmp = R2stmp(respthind2);
    R2stotinc = [R2stotinc, R2stmp];
    sigmastotinc = [sigmastotinc, sigmastmp];
    
    sigmastmp = sigmas_layers_log{lind, 1};
    sigmastotdeclog = [sigmastotdeclog, sigmastmp];
    
    sigmastmp = sigmas_layers_log{lind, 2};
    sigmastotinclog = [sigmastotinclog, sigmastmp];
    toc
end
thtmp = 0.90;
indtmp1 = R2stotdec>thtmp;
indtmp2 = R2stotinc>thtmp;
sigmastotdec = sigmastotdec(indtmp1);
sigmastotinc = sigmastotinc(indtmp2);

% figure
% hold on;
% histogram(sigmastotdec(indtmp1))
% mean(sigmastotdec(indtmp1))
% histogram(sigmastotinc(indtmp2))
% mean(sigmastotinc(indtmp2))



