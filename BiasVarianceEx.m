x = linspace(-1, 1, 200);   % for the plots
fx = @(x) (sin(pi * x));    % function to approximate
n = 1;                      % CHANGE IT
d = 1;                      % CHANGE IT
g1 = @(a, x) (a*x);         % function estimating fx
g2 = @(b, x) (b*ones(1, length(x)));
g3 = @(a, b, x) (a*x + b);
g4 = @(a, x) (a*x.^2);
g5 = @(a, b, x) (a*x.^2 + b);
doPlot = false;
% simulation 
interval = [-1, 1];
numPts = 2;
N = 100000;
a = zeros(1, N);
b = zeros(1, N);
varTmp = zeros(1, length(x));


%% First model y = ax
if doPlot
  figure(1)
  plot(x, fx(x))
  hold on
  grid on
  axis([-1 1 -1 1])
end

for k = 1:N
  pts = interval(1) + diff(interval) .* rand(numPts,1);
  LE = LinearRegressor(1);
  LE = LE.train(pts, fx(pts));
  a(k) = LE.w;
  if doPlot
    line = a(k) .* x;
    h = plot(x, line, 'r', 'LineWidth', 0.5);
    h.Color = [1 0 0 0.01];
  end
end
expA = mean(a);
gbar = g1(expA, x);
if doPlot
  plot(x, gbar, 'k', 'LineWidth', 3);
end
% Bias
bias = mean((gbar - fx(x)).^2 );
% Variance

for m = 1:length(x)
  varTmp(m) = mean((g1(a(m), x) - gbar).^2);
end
variance = mean(varTmp);

E_out = bias + variance;
disp(['Bias = ' num2str(bias) ' -- Variance = ' num2str(variance) ' -- E_out for model y=ax : ' num2str(E_out)])

%% Second model y = b
if doPlot
  figure(2)
  plot(x, fx(x))
  hold on
  grid on
  axis([-1 1 -1 1])
end

for k = 1:N
  pts = interval(1) + diff(interval) .* rand(numPts,1);
  b(k) = sum(sin(pi*pts)) / 2;
  if doPlot
    line = b(k) .* ones(1, length(x));
    h = plot(x, line, 'r', 'LineWidth', 0.5);
    h.Color = [1 0 0 0.01];
  end
end
expB = mean(b);
gbar = g2(expB, x);
if doPlot
  plot(x, gbar, 'k', 'LineWidth', 3);
end
% Bias
bias = mean((gbar - fx(x)).^2 );
% Variance
for m = 1:length(x)
  varTmp(m) = mean((g2(b(m), x) - gbar).^2);
end
variance = mean(varTmp);

E_out = bias + variance;
disp(['Bias = ' num2str(bias) ' -- Variance = ' num2str(variance) ' -- E_out for model y = b : ' num2str(E_out)])


%% Third model y = ax + b
if doPlot
  figure(3)
  plot(x, fx(x))
  hold on
  grid on
  axis([-1 1 -1 1])
end

for k = 1:N
  pts = interval(1) + diff(interval) .* rand(numPts,1);
  trainPts = [ones(2, 1), pts];
  LE = LinearRegressor(2);
  LE = LE.train(trainPts, fx(pts));
  b(k) = LE.w(1);
  a(k) = LE.w(2);
  if doPlot
    line = g3(a(k), b(k), x);
    h = plot(x, line, 'r', 'LineWidth', 0.5);
    h.Color = [1 0 0 0.01];
  end
end

expA = mean(a);
expB = mean(b);
gbar = g3(expA, expB, x);
if doPlot
  plot(x, gbar, 'k', 'LineWidth', 3);
end
% Bias
bias = mean((gbar - fx(x)).^2 );
% Variance
for m = 1:length(x)
  varTmp(m) = mean((g3(a(m), b(m), x) - gbar).^2);
end
variance = mean(varTmp);

E_out = bias + variance;
disp(['Bias = ' num2str(bias) ' -- Variance = ' num2str(variance) ' -- E_out for model y = ax + b : ' num2str(E_out)])


%% Fourth model y = ax^2
if doPlot
  figure(4)
  plot(x, fx(x))
  hold on
  grid on
  axis([-1 1 -1 1])
end

for k = 1:N
  pts = interval(1) + diff(interval) .* rand(numPts,1);
  pts = pts .^2;
  LE = LinearRegressor(1);
  LE = LE.train(pts, fx(pts));
  a(k) = LE.w;
  if doPlot
    parabola = g4(a(k), x);
    h = plot(x, parabola, 'r', 'LineWidth', 0.5);
    h.Color = [1 0 0 0.01];
  end
end
expA = mean(a);
gbar = g4(expA, x);
if doPlot
  plot(x, gbar, 'k', 'LineWidth', 3);
end
% Bias
bias = mean((gbar - fx(x)).^2 );
% Variance

for m = 1:length(x)
  varTmp(m) = mean((g4(a(m), x) - gbar).^2);
end
variance = mean(varTmp);

E_out = bias + variance;
disp(['Bias = ' num2str(bias) ' -- Variance = ' num2str(variance) ' -- E_out for model y = ax^2: ' num2str(E_out)])

%% Fifth model y = ax^2 + b
if doPlot
  figure(5)
  plot(x, fx(x))
  hold on
  grid on
  axis([-1 1 -1 1])
end

for k = 1:N
  pts = interval(1) + diff(interval) .* rand(numPts,1);
  pts = pts .^2;
  trainPts = [ones(2, 1), pts];
  LE = LinearRegressor(2);
  LE = LE.train(trainPts, fx(pts));
  b(k) = LE.w(1);
  a(k) = LE.w(2);
  if doPlot
    parabola = g5(a(k), b(k), x);
    h = plot(x, parabola, 'r', 'LineWidth', 0.5);
    h.Color = [1 0 0 0.01];
  end
end

expA = mean(a);
expB = mean(b);
gbar = g5(expA, expB, x);
if doPlot
  plot(x, gbar, 'k', 'LineWidth', 3);
end
% Bias
bias = mean((gbar - fx(x)).^2 );
% Variance
for m = 1:length(x)
  varTmp(m) = mean((g5(a(m), b(m), x) - gbar).^2);
end
variance = mean(varTmp);


E_out = bias + variance;
disp(['Bias = ' num2str(bias) ' -- Variance = ' num2str(variance) ' -- E_out for model y = ax^2 + b : ' num2str(E_out)])
