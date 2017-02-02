numExp = 100000;

numCoins = 1000;
numLaunches = 10;

vRand = zeros(numExp, 1);
vOne = zeros(numExp, 1);
vMin = zeros(numExp, 1);

for k = 1:numExp
  coins = randi([0, 1], numCoins, numLaunches);
  %choose random position
  ranpos = randi(numCoins, 1);
  vRand(k) = sum(coins(ranpos, :)) / numLaunches;
  %choose first coin
  vOne(k) = sum(coins(1, :)) / numLaunches;
  %choose minimum
  coinSum = sum(coins, 2);
  [~, minPos] = min(coinSum);
  vMin(k) = sum(coins(minPos, :)) / numLaunches;
end

disp(mean(vMin));