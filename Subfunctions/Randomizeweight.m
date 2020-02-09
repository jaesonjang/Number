function net_test = Randomizeweight_permute(net, rand_layers_ind)
net_test = net;
net_tmp = net_test.saveobj;
for ind_tl = 1:length(rand_layers_ind)
    % rand_layers_ind = [2, 6, 10, 12 14];
    % ind_tl = 1;
    % LOI = layers_set{ind_tl};
    targetlayer_ind = rand_layers_ind(ind_tl);
    weight_conv = net.Layers(targetlayer_ind ,1).Weights;
    bias_conv = net.Layers(targetlayer_ind ,1).Bias;
    wstd = std(weight_conv(:));
    bstd = std(bias_conv(:));
    wmean = mean(weight_conv(:));
    bmean = mean(bias_conv(:));
    
    %% change network parameters
    % num_NS = zeros(1,10);
    % for iii = 1:10
    weight_conv_randomize =wmean+wstd*randn(size(weight_conv));
    bias_conv_randomize = bmean+bstd*randn(size(bias_conv));
    
    net_tmp.Layers(targetlayer_ind).Weights = weight_conv_randomize;
    net_tmp.Layers(targetlayer_ind).Bias = bias_conv_randomize;
end
net_test = net_test.loadobj(net_tmp);
% figure
% imagesc(net.Layers(2).Weights(:,:,1,1))
%
% figure
% imagesc(net_test.Layers(2).Weights(:,:,1,1))

end