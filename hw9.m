%% SPECIFY CONSTANTS
DIM = 2;
INTERVAL = [-1 1];
LAMBDA = 1;

%% READ DATA
fid = fopen('../features.train');
in = fscanf(fid, '%f %f %f');
in = reshape(in, 3, length(in)/3);
in = in';
fclose(fid);

fid = fopen('../features.test');
out = fscanf(fid, '%f %f %f');
out = reshape(out, 3, length(out)/3);
out = out';
fclose(fid);

%% QUESTION 7
vals = 5:9;
E_in_ex7 = zeros(1, length(vals));
for k = 1:length(vals);
  ex7 = in;
  ex7(ex7(:,1) ~= vals(k), 1) = -1;
  ex7(ex7(:,1) == vals(k), 1) = +1;
  linReg = LinearRegression(DIM, LAMBDA);
  linReg = linReg.train(ex7(:, 2:3), ex7(:, 1));
  yhat = linReg.classify(ex7(:, 2:3));
  E_in_ex7(k) = sum(yhat ~= ex7(:, 1)) / length(ex7);
end

%% QUESTION 8
vals = 0:4;
E_out_ex8 = zeros(1, length(vals));
NEWDIM = 6;
for k = 1:length(vals);
  %train
  ex8 = in;
  ex8(ex8(:,1) ~= vals(k), 1) = -1;
  ex8(ex8(:,1) == vals(k), 1) = +1;
  y = ex8(:,1);
  X = ex8(:, 2:3);
  Z = ones(length(X), NEWDIM);
  Z(:, 2:3) = X;
  Z(:, 4) = X(:,1) .* X(:,2);
  Z(:, 5) = X(:,1) .^ 2;
  Z(:, 6) = X(:,2) .^ 2;
  
  %test
  ex8_test = out;
  ex8_test(ex8_test(:,1) ~= vals(k), 1) = -1;
  ex8_test(ex8_test(:,1) == vals(k), 1) = +1;
  y_test = ex8_test(:,1);
  X_test = ex8_test(:, 2:3);
  Z_test = ones(length(X_test), NEWDIM);
  Z_test(:, 2:3) = X_test;
  Z_test(:, 4) = X_test(:,1) .* X_test(:,2);
  Z_test(:, 5) = X_test(:,1) .^ 2;
  Z_test(:, 6) = X_test(:,2) .^ 2;

  %linear regression
  linReg = LinearRegression(NEWDIM, LAMBDA);
  linReg = linReg.train(Z, y);
  yhat = linReg.classify(Z_test);
  E_out_ex8(k) = sum(yhat ~= y_test) / length(y_test);
end

%% QUESTION 9
vals = 0:9;
E_in_ex9 = zeros(1, length(vals));
E_out_ex9 = zeros(1, length(vals));
E_in_ex9_tr = zeros(1, length(vals));
E_out_ex9_tr = zeros(1, length(vals));
NEWDIM = 6;
for k = 1:length(vals);
  %train
  ex9_train = in;
  ex9_train(ex9_train(:,1) ~= vals(k), 1) = -1;
  ex9_train(ex9_train(:,1) == vals(k), 1) = +1;
  y = ex9_train(:,1);
  X = ex9_train(:, 2:3);
  Z = ones(length(X), NEWDIM);
  Z(:, 2:3) = X;
  Z(:, 4) = X(:,1) .* X(:,2);
  Z(:, 5) = X(:,1) .^ 2;
  Z(:, 6) = X(:,2) .^ 2;
  
  %test
  ex9_test = out;
  ex9_test(ex9_test(:,1) ~= vals(k), 1) = -1;
  ex9_test(ex9_test(:,1) == vals(k), 1) = +1;
  y_test = ex9_test(:,1);
  X_test = ex9_test(:, 2:3);
  Z_test = ones(length(X_test), NEWDIM);
  Z_test(:, 2:3) = X_test;
  Z_test(:, 4) = X_test(:,1) .* X_test(:,2);
  Z_test(:, 5) = X_test(:,1) .^ 2;
  Z_test(:, 6) = X_test(:,2) .^ 2;

  %linear regression on input
  linReg = LinearRegression(DIM, LAMBDA);
  linReg = linReg.train(X, y);
  yhat_train = linReg.classify(X);
  yhat_test = linReg.classify(X_test);
  E_in_ex9(k) = sum(yhat_train ~= y) / length(y);
  E_out_ex9(k) = sum(yhat_test ~= y_test) / length(y_test);
  
  %linear regression on transformed data
  linReg = LinearRegression(NEWDIM, LAMBDA);
  linReg = linReg.train(Z, y);
  yhat_train = linReg.classify(Z);
  yhat_test = linReg.classify(Z_test);
  E_in_ex9_tr(k) = sum(yhat_train ~= y) / length(y);
  E_out_ex9_tr(k) = sum(yhat_test ~= y_test) / length(y_test);
end

%% QUESTION 10
lambda = [0.01 1];
NEWDIM = 6;
ex10_train = in;
ex10_train(ex10_train(:,1) ~= 1 & ex10_train(:,1) ~= 5, :) = [];
ex10_train(ex10_train(:,1) == 5, 1) = -1;

ex10_test = out;
ex10_test(ex10_test(:,1) ~= 1 & ex10_test(:,1) ~= 5, :) = [];
ex10_test(ex10_test(:,1) == 5, 1) = -1;

y_train = ex10_train(:,1);
X_train = ex10_train(:, 2:3);
Z_train = ones(length(X_train), NEWDIM);
Z_train(:, 2:3) = X_train;
Z_train(:, 4) = X_train(:,1) .* X_train(:,2);
Z_train(:, 5) = X_train(:,1) .^ 2;
Z_train(:, 6) = X_train(:,2) .^ 2;

y_test = ex10_test(:, 1);
X_test = ex10_test(:, 2:3);
Z_test = ones(length(X_test), NEWDIM);
Z_test(:, 2:3) = X_test;
Z_test(:, 4) = X_test(:,1) .* X_test(:,2);
Z_test(:, 5) = X_test(:,1) .^ 2;
Z_test(:, 6) = X_test(:,2) .^ 2;

E_in_ex10 = zeros(1, length(lambda));
E_out_ex10 = zeros(1, length(lambda));

for k = 1:length(lambda)
  linReg = LinearRegression(NEWDIM, lambda(k));
  linReg = linReg.train(Z_train, y_train);
  yhat_train = linReg.classify(Z_train);
  yhat_test = linReg.classify(Z_test);
  E_in_ex10(k) = sum(yhat_train ~= y_train) / length(y_train);
  E_out_ex10(k) = sum(yhat_test ~= y_test) / length(y_test);
end

%% QUESTION 11
X = [1, 0; 0, 1; 0, -1; -1, 0; 0, -2; 0, 2; -2, 0];
y = [-1 -1 -1 1 1 1 1];
figure(1)
gscatter(X(:,1), X(:,2), y);
grid on

Z1 = X(:,2).^2 - 2*X(:,1) - 1;
Z2 = X(:,1).^2 - 2*X(:,2) + 1;
Z = [Z1 Z2];
figure(2)
gscatter(Z(:,1), Z(:,2), y);
grid on

%% QUESTION 13 - 14
f = @(x1, x2) sign(x2 - x1 + 0.25 * sin(pi*x1));
NUMPTS = 100;
doPlots = false;
lambda = 1.5;
NUM_RUNS = 10000;
E_in_ex13 = zeros(1, NUM_RUNS);
E_out_ex13 = zeros(1, NUM_RUNS);
E_in_ex14 = zeros(2, NUM_RUNS);
E_out_ex14 = zeros(2, NUM_RUNS);
NUM_CLUSTERS = [9, 12];
nonValidIter = 0;
for k = 1:NUM_RUNS
  %generate points
  X = INTERVAL(1) + diff(INTERVAL)*rand(NUMPTS, DIM);
  X_test = INTERVAL(1) + diff(INTERVAL)*rand(NUMPTS, DIM);
  y = f(X(:,1), X(:,2));
  y_test = f(X_test(:,1), X_test(:,2));
  if doPlots
    figure(k)
    hold on
    axis([INTERVAL(1) INTERVAL(2) INTERVAL(1) INTERVAL(2)])
    gscatter(X(:,1), X(:,2), y)
  end
  % do svm train
  opts = ['-s 0 -t 2 -g ' num2str(lambda) ' -c 1e10 -q'];
  model = svmtrain(y, X, opts);
  [~, accuracy_in, ~] = svmpredict(y, X, model);
  E_in_ex13(k) = (100 - (accuracy_in(1))) / 100;
  [~, accuracy_out, ~] = svmpredict(y_test, X_test, model);
  E_out_ex13(k) = (100 - (accuracy_out(1))) / 100;
  
  for l = 1:length(NUM_CLUSTERS)
    % do Kmeans
    km = clustering.FastKMeans(X, NUM_CLUSTERS(l));
    km = km.fit;
    labels = km.getLabels;
    cluCenters = km.getCenters;

    %ignore cases of empty clusters
    if length(unique(labels)) == NUM_CLUSTERS(l)
      phi = ones(NUMPTS, 1 + NUM_CLUSTERS(l));
      for m = 1:NUM_CLUSTERS(l)
        for n = 1:NUMPTS
          phi(n, 1+m) = exp(-lambda * sqrt((X(n, 1) - cluCenters(m, 1)).^2 + (X(n, 2) - cluCenters(m, 2)).^2));
        end
      end
      w = pinv(phi) * y;

      %E_in
      yhat = zeros(length(y), 1);
      for n = 1:NUMPTS
        yhat(n) = sign(w(1) + phi(n, 2:end) * w(2:end));
      end
      E_in_ex14(l, k) = sum(yhat ~= y) / length(yhat);

      %E_out
      yhat = zeros(length(y), 1);
      phi = zeros(1, NUM_CLUSTERS(l));
      for m = 1:NUM_CLUSTERS(l)
        for n = 1:NUMPTS
          phi(n, 1+m) = exp(-lambda * sqrt((X_test(n, 1) - cluCenters(m, 1)).^2 + (X_test(n, 2) - cluCenters(m, 2)).^2));
        end
      end
      for n = 1:NUMPTS
        yhat(n) = sign(w(1) + phi(n, 2:end) * w(2:end));
      end

      E_out_ex14(l, k) = sum(yhat ~= y_test) / length(yhat);
    else
      nonValidIter = nonValidIter +1;
    end
  end
end

%% QUESTION 16
f = @(x1, x2) sign(x2 - x1 + 0.25 * sin(pi*x1));
NUMPTS = 100;
lambda = 1.5;
NUM_RUNS = 10000;
NUM_CLUSTERS = [9, 12];
E_in_ex16 = zeros(2, NUM_RUNS);
E_out_ex16 = zeros(2, NUM_RUNS);
nonValidIter = 0;
for k = 1:NUM_RUNS
  %generate points
  X = INTERVAL(1) + diff(INTERVAL)*rand(NUMPTS, DIM);
  X_test = INTERVAL(1) + diff(INTERVAL)*rand(NUMPTS, DIM);
  y = f(X(:,1), X(:,2));
  y_test = f(X_test(:,1), X_test(:,2));
  
  for l = 1:length(NUM_CLUSTERS)
    % do Kmeans
  km = clustering.FastKMeans(X, NUM_CLUSTERS(l));
  km = km.fit;
  labels = km.getLabels;
  cluCenters = km.getCenters;
  
  %ignore cases of empty clusters
  if length(unique(labels)) == NUM_CLUSTERS(l)
    phi = ones(NUMPTS, 1 + NUM_CLUSTERS(l));
    for m = 1:NUM_CLUSTERS(l)
      for n = 1:NUMPTS
        phi(n, 1+m) = exp(-lambda * sqrt((X(n, 1) - cluCenters(m, 1)).^2 + (X(n, 2) - cluCenters(m, 2)).^2));
      end
    end
    w = pinv(phi) * y;
    
    %E_in
    yhat = zeros(length(y), 1);
    for n = 1:NUMPTS
      yhat(n) = sign(w(1) + phi(n, 2:end) * w(2:end));
    end
    E_in_ex16(l, k) = sum(yhat ~= y) / length(yhat);
    
    %E_out
    yhat = zeros(length(y), 1);
    phi = zeros(1, NUM_CLUSTERS(l));
    for m = 1:NUM_CLUSTERS(l)
      for n = 1:NUMPTS
        phi(n, 1+m) = exp(-lambda * sqrt((X_test(n, 1) - cluCenters(m, 1)).^2 + (X_test(n, 2) - cluCenters(m, 2)).^2));
      end
    end
    for n = 1:NUMPTS
      yhat(n) = sign(w(1) + phi(n, 2:end) * w(2:end));
    end

    E_out_ex16(l, k) = sum(yhat ~= y_test) / length(yhat);
    else
      nonValidIter = nonValidIter +1;
    end
  end
  
end

%% QUESTION 17
f = @(x1, x2) sign(x2 - x1 + 0.25 * sin(pi*x1));
NUMPTS = 100;
lambda = [1.5 2];
NUM_RUNS = 10000;
NUM_CLUSTERS = 9;
E_in_ex17 = zeros(2, NUM_RUNS);
E_out_ex17 = zeros(2, NUM_RUNS);
nonValidIter = 0;
for k = 1:NUM_RUNS
  %generate points
  X = INTERVAL(1) + diff(INTERVAL)*rand(NUMPTS, DIM);
  X_test = INTERVAL(1) + diff(INTERVAL)*rand(NUMPTS, DIM);
  y = f(X(:,1), X(:,2));
  y_test = f(X_test(:,1), X_test(:,2));
  
  for l = 1:length(lambda)
    % do Kmeans
  km = clustering.FastKMeans(X, NUM_CLUSTERS);
  km = km.fit;
  labels = km.getLabels;
  cluCenters = km.getCenters;
  
  %ignore cases of empty clusters
  if length(unique(labels)) == NUM_CLUSTERS
    phi = ones(NUMPTS, 1 + NUM_CLUSTERS);
    for m = 1:NUM_CLUSTERS
      for n = 1:NUMPTS
        phi(n, 1+m) = exp(-lambda(l) * sqrt((X(n, 1) - cluCenters(m, 1)).^2 + (X(n, 2) - cluCenters(m, 2)).^2));
      end
    end
    w = pinv(phi) * y;
    
    %E_in
    yhat = zeros(length(y), 1);
    for n = 1:NUMPTS
      yhat(n) = sign(w(1) + phi(n, 2:end) * w(2:end));
    end
    E_in_ex17(l, k) = sum(yhat ~= y) / length(yhat);
    
    %E_out
    yhat = zeros(length(y), 1);
    phi = zeros(1, NUM_CLUSTERS);
    for m = 1:NUM_CLUSTERS
      for n = 1:NUMPTS
        phi(n, 1+m) = exp(-lambda(l) * sqrt((X_test(n, 1) - cluCenters(m, 1)).^2 + (X_test(n, 2) - cluCenters(m, 2)).^2));
      end
    end
    for n = 1:NUMPTS
      yhat(n) = sign(w(1) + phi(n, 2:end) * w(2:end));
    end

    E_out_ex17(l, k) = sum(yhat ~= y_test) / length(yhat);
    else
      nonValidIter = nonValidIter +1;
    end
  end
  
end