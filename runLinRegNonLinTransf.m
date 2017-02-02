%% SPECIFY CONSTANTS
DIM = 2;
INTERVAL = [-1 1];
NUMPTS = 1000;
NUM_REPEAT = 1000;
numIt = zeros(1, NUM_REPEAT);
disag = zeros(1, NUM_REPEAT);
doPlots = true;
plotX = linspace(INTERVAL(1), INTERVAL(2), 100);
hypotheses = zeros(1 + DIM, NUM_REPEAT);
Ein = zeros(NUM_REPEAT, 1);
EinNew = zeros(NUM_REPEAT, 1);
Eout = zeros(NUM_REPEAT, 1);
wnew = zeros(NUM_REPEAT, 6);

for kk = 1:NUM_REPEAT

  %% EXTRACT NUMPTS POINTS
  X = ones(NUMPTS, 1+DIM);
  X(:, 2:1+DIM) = INTERVAL(1) + (INTERVAL(2) - INTERVAL(1)) .* rand(NUMPTS, DIM);
  y = sign(X(:,3).^2 + X(:,2).^2 - 0.6);
  
  %% NOW FLIP 10% of input
  wrongPts = round(0.1*NUMPTS);
  posWrong = datasample(1:NUMPTS, wrongPts);
  y(posWrong) = -1 * y(posWrong);

  %% RUN Linear regression
  linReg = LinearModel(1 + DIM);
  linReg = linReg.train(X, y);
  yhat = linReg.classify(X);
  Ein(kk) = sum(yhat ~= y) / numel(y);

  %% Transform data
  newNumDim = 6;
  Xnew = ones(NUMPTS, newNumDim);
  Xnew(:, 1:3) = X;
  Xnew(:, 4) = X(:,2) .* X(:,3);
  Xnew(:, 5) = X(:,2) .^ 2;
  Xnew(:, 6) = X(:,3) .^ 2;
  
  %% RUN Linear regression on transformed data
  linReg2 = LinearRegression(newNumDim);
  linReg2 = linReg2.train(Xnew, y);
  wnew(kk, :) = linReg2.w';
  yhat = linReg2.classify(Xnew);
  EinNew(kk) = sum(yhat ~= y) / numel(y);
  
  %% GENERATE TEST SET
  test = ones(NUMPTS, 1 + DIM);
  test(:, 2:end) = INTERVAL(1) + (INTERVAL(2) - INTERVAL(1)) .* rand(1000, 2);
  testnew = ones(NUMPTS, newNumDim);
  testnew(:, 1:3) = test;
  testnew(:, 4) = test(:,2) .* test(:,3);
  testnew(:, 5) = test(:,2) .^ 2;
  testnew(:, 6) = test(:,3) .^ 2;
  yNew = sign(test(:,3).^2 + test(:,2).^2 - 0.6);
  
  %% NOW FLIP 10% of input
  wrongPts = round(0.1*NUMPTS);
  posWrong = datasample(1:NUMPTS, wrongPts);
  yNew(posWrong) = -1 * yNew(posWrong);
  
  %%CLASSIFY AND CHECK ERROR
  yHatNew = linReg2.classify(testnew);
  Eout(kk) = sum(yHatNew ~= yNew) / numel(yNew);
  
end

disp(mean(Ein))
disp(mean(EinNew))
disp(mean(Eout))
disp(mean(wnew, 1))