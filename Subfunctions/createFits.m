function [fitresult, gof] = createFits(WV_part, sharp_part, sigs_egs, ratio_tot, performance_tot)
%CREATEFITS1(WV_PART,SHARP_PART,SIGS_EGS,RATIO_TOT,PERFORMANCE_TOT)
%  Create fits.
%
%  Data for 'sharpness_sigmoid' fit:
%      X Input : WV_part
%      Y Output: sharp_part
%  Data for 'ratio_sigmoid' fit:
%      X Input : sigs_egs
%      Y Output: ratio_tot
%  Data for 'performance_sigmoid' fit:
%      X Input : sigs_egs
%      Y Output: performance_tot
%  Output:
%      fitresult : a cell-array of fit objects representing the fits.
%      gof : structure array with goodness-of fit info.
%
%  참고 항목 FIT, CFIT, SFIT.

%  MATLAB에서 12-Nov-2019 12:57:12에 자동 생성됨

%% Initialization.

% Initialize arrays to store fits and goodness-of-fit.
fitresult = cell( 3, 1 );
gof = struct( 'sse', cell( 3, 1 ), ...
    'rsquare', [], 'dfe', [], 'adjrsquare', [], 'rmse', [] );

%% Fit: 'sharpness_sigmoid'.
[xData, yData] = prepareCurveData( WV_part, sharp_part );

% Set up fittype and options.
 ft = fittype( 'a/(1+b*exp(-c*x))', 'independent', 'x', 'dependent', 'y' );
%ft = fittype( 'a/(1+exp(-c*(x-b)))', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0.678735154857773 0.757740130578333 0.743132468124916];

% Fit model to data.
[fitresult{1}, gof(1)] = fit( xData, yData, ft, opts );

% Plot fit with data.
figure( 'Name', 'sharpness_sigmoid' );
h = plot( fitresult{1}, xData, yData );
legend( h, 'sharp_part vs. WV_part', 'sharpness_sigmoid', 'Location', 'NorthEast' );
% Label axes
xlabel WV_part
ylabel sharp_part
grid on

%% Fit: 'ratio_sigmoid'.
[xData, yData] = prepareCurveData( sigs_egs, ratio_tot );

% Set up fittype and options.
 ft = fittype( 'a/(1+b*exp(-c*x))', 'independent', 'x', 'dependent', 'y' );
%ft = fittype( 'a/(1+exp(-c*(x-b)))', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0.691883707364558 0.389426266707162 0.707661521370473];

% Fit model to data.
[fitresult{2}, gof(2)] = fit( xData, yData, ft, opts );

% Plot fit with data.
figure( 'Name', 'ratio_sigmoid' );
h = plot( fitresult{2}, xData, yData );
legend( h, 'ratio_tot vs. sigs_egs', 'ratio_sigmoid', 'Location', 'NorthEast' );
% Label axes
xlabel sigs_egs
ylabel ratio_tot
grid on

%% Fit: 'performance_sigmoid'.
[xData, yData] = prepareCurveData( sigs_egs, performance_tot );

% Set up fittype and options.
 ft = fittype( '0.5+a/(1+b*exp(-c*x))', 'independent', 'x', 'dependent', 'y' );
%ft = fittype( '0.5+ a/(1+exp(-c*(x-b)))', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0.746629428031745 0.381718241879865 0.637359884152111];

% Fit model to data.
[fitresult{3}, gof(3)] = fit( xData, yData, ft, opts );

% Plot fit with data.
figure( 'Name', 'performance_sigmoid' );
h = plot( fitresult{3}, xData, yData );
legend( h, 'performance_tot vs. sigs_egs', 'performance_sigmoid', 'Location', 'NorthEast' );
% Label axes
xlabel sigs_egs
ylabel performance_tot
grid on


