function [response_tot] = getactivation(net, LOI, image_sets)


%% get size of the image
imtmp = squeeze(image_sets(:,:, 1,1));
imtmpp = imtmp*255;
im = cat(3, imtmpp , imtmpp , imtmpp );
% imgSize = size(im);
% imgSize = imgSize(1:2);
act = activations(net,im,LOI);
acttmp = act(:);
N_neurons = length(acttmp);



number_N = size(image_sets,4);
image_iter = size(image_sets,3);


response_tot = zeros(number_N, image_iter, N_neurons);
for ii = 1:number_N
for jj = 1:image_iter
imtmp = squeeze(image_sets(:,:,jj,ii));
imtmpp = imtmp*255;
im = cat(3, imtmpp , imtmpp , imtmpp );
imgSize = size(im);
imgSize = imgSize(1:2);
act = activations(net,im,LOI);
acttmp = act(:);
response_tot(ii,jj,:) = acttmp;
end
end

end