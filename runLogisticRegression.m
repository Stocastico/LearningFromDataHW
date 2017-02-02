%% SPECIFY CONSTANTS
DIM = 2;
INTERVAL = [-1 1];
NUMPTS = 100;
NUM_REPEAT = 100;
numIt = zeros(1, NUM_REPEAT);
disag = zeros(1, NUM_REPEAT);
doPlots = false;
plotX = linspace(INTERVAL(1), INTERVAL(2), 100);
hypotheses = zeros(1 + DIM, NUM_REPEAT);
Ein = zeros(NUM_REPEAT, 1);
Eout = zeros(NUM_REPEAT, 1);
epochs = zeros(NUM_REPEAT, 1);
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
  
  %% RUN LOGISTIC REGRESSION
  logReg = LogisticRegression(1 + DIM);
  [logReg, epochs(kk)] = logReg.trainEx8(X, y);
  %hypotheses(:, kk) = logReg.w;
  %yhat = logReg.predict(X);
  Ein(kk) = logReg.calcError(X, y);
  
  %% COMPUTE OUT OF SAMPLE ERROR
  numPtsTest = 10000;
  XNew = ones(numPtsTest, 1 + DIM);
  XNew(:, 2:end) = INTERVAL(1) + (INTERVAL(2) - INTERVAL(1)) .* rand(numPtsTest, 2);
  yNew = double(XNew(:, 3) - m * XNew(:, 2) - q > 0);
  yNew(yNew == 0) = -1;
  Eout(kk) = logReg.calcError(XNew, yNew);
  
  if doPlots
    plotY = (-logReg.w(1) / logReg.w(3)) + (-logReg.w(2) / logReg.w(3))*plotX;
    plot(plotX, plotY, 'r');
    close(1)
  end
  
end

mean(Ein)
mean(epochs)
mean(Eout)