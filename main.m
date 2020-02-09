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
% =========================================================================================================================================================
clear
close all

load('flg.mat')

%%
disp('Demo codes for "Spontaneous generation of number sense in untrained deep neural networks')
disp(' ')
disp('* Select figure numbers (from 1 to 4) that you want to perform a demo simulation.')
disp('* It performs a demo version (a fewer set of stimuli than in the paper) of simulation using a single condition of the network.')
disp('  (# images: 2400 -> 480, # repetition of simulation: 100 (or 1000 for summation model) -> 1)')
disp('* Expected running time is about 5 minutes for each figure, but may vary by system conditions.')
disp(' ')

simulation_option = 1;

color_linear = [241 90 36]/255;
color_log = [0 146 69]/255;

%%
toolbox_chk

if flg1
    clearvars -except simulation_option color_linear color_log flg1 flg2 flg3 flg4
    disp('Demo results for Fig. 1 will be displayed...')
    Figure1
end

if flg2
    clearvars -except simulation_option color_linear color_log flg1 flg2 flg3 flg4
    disp('Demo results for Fig. 2 will be displayed...')
    Figure2
end

if flg3
    clearvars -except simulation_option color_linear color_log flg1 flg2 flg3 flg4
    disp('Demo results for Fig. 3 will be displayed...')
    Figure3
end

if flg4
    clearvars -except simulation_option color_linear color_log flg1 flg2 flg3 flg4
    disp('Demo results for Fig. 4 will be displayed...')
    Figure4
end