%% SPECIFY CONSTANTS
DIM = 2;
INTERVAL = [-1 1];
NUMPTS = 10;
NUM_REPEAT = 1000;
numIt = zeros(1, NUM_REPEAT);
disag = zeros(1, NUM_REPEAT);
doPlots = false;
plotX = linspace(INTERVAL(1), INTERVAL(2), 100);
hypotheses = zeros(1 + DIM, NUM_REPEAT);
Ein = zeros(NUM_REPEAT, 1);
Eout = zeros(NUM_REPEAT, 1);
iter = zeros(NUM_REPEAT, 1);

for kk = 1:NUM_REPEAT

  %% DEFINE A LINE
  linePts = INTERVAL(1) + (INTERVAL(2) - INTERVAL(1)) .* rand(2, 2);
  %lineequation =  A + Bx + Cy = 0; V = (A, B, C)
  V = [linePts(2,1)*linePts(1,2) - linePts(1,1)*linePts(2,2), linePts(2, 2) - linePts(1, 2), linePts(1, 1) - linePts(2, 1)];
  m = -V(2) / V(3);
  q = -V(1) / V(3);
  plotY = q + m*plotX;

  %% EXTRACT NUMPTS POINTS
  X = ones(NUMPTS, 1+DIM);
  X(:, 2:1+DIM) = INTERVAL(1) + (INTERVAL(2) - INTERVAL(1)) .* rand(NUMPTS, DIM);
  y = double(X(:,3) - m * X(:,2) - q > 0);
  y(y == 0) = -1;

  %% PLOT DATASET
  if doPlots
    figure(1)
    gscatter(X(:,2), X(:,3), y)
    hold on
    plot(plotX, plotY, 'k');
    axis([INTERVAL(1) INTERVAL(2) INTERVAL(1) INTERVAL(2)])
  end

  %% RUN Linear regression
  linReg = LinearRegression(1 + DIM);
  linReg = linReg.train(X, y);
  hypotheses(:, kk) = linReg.w;
  yhat = linReg.classify(X);
  Ein(kk) = sum(yhat ~= y) / numel(y);
  
  %% COMPUTE OUT OF SAMPLE ERROR
  XNew = ones(1000, 1 + DIM);
  XNew(:, 2:end) = INTERVAL(1) + (INTERVAL(2) - INTERVAL(1)) .* rand(1000, 2);
  yNew = double(XNew(:, 3) - m * XNew(:, 2) - q > 0);
  yNew(yNew == 0) = -1;
  yHatNew = linReg.classify(XNew);
  Eout(kk) = sum(yHatNew ~= yNew) / numel(yNew);
  
  %% NOW LEARN PERCEPTRON
  %transpose X
  X = X';
  
  w = linReg.w;
  pla = PLA(1+DIM, w, false);
  pla = pla.train(X, y);
  iter(kk) = pla.numIt;
  
  if doPlots
    plotY = (-linReg.w(1) / linReg.w(3)) + (-linReg.w(2) / linReg.w(3))*plotX;
    plot(plotX, plotY, 'r');
    close(1)
  end
end

mean(Ein)
mean(Eout)
mean(iter)