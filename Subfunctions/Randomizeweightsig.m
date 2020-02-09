function net_test = Randomizeweightsig(net, layers_ind, sig)
net_test = net;
net_tmp = net_test.saveobj;
for ind_tl = 1:length(layers_ind)
    % rand_layers_ind = [2, 6, 10, 12 14];
    % ind_tl = 1;
    % LOI = layers_set{ind_tl};
    targetlayer_ind = layers_ind(ind_tl);
    
    weight_conv = net.Layers(targetlayer_ind ,1).Weights;
    bias_conv = net.Layers(targetlayer_ind ,1).Bias;
    
    % figure
    % subplot(1,2,1)
    % histogram(weight_conv4(:))
    % subplot(1,2,2)
    % histogram(bias_conv4(:))
    wstd = sig*std(weight_conv(:));
    bstd = sig*std(bias_conv(:));
    wmean = mean(weight_conv(:));
    bmean = mean(bias_conv(:));
    
    %% change network parameters
    % num_NS = zeros(1,10);
    % for iii = 1:10
    weight_conv_randomize =wmean+wstd*randn(size(weight_conv));
    bias_conv_randomize = bmean+bstd*randn(size(bias_conv));
    net_tmp.Layers(targetlayer_ind ,1).Weights = weight_conv_randomize;
    net_tmp.Layers(targetlayer_ind ,1).Bias = bias_conv_randomize;
    
    
end
net_test = net_test.loadobj(net_tmp);
end