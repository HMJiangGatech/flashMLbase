rng shuffle

n = 200;
d = 2000;
nr = 2;
sigma = 1;
var = eye(d)*0.5 + 0.5;
mu = randn(1,d)*0;
beta = zeros(d,nr);
beta(1,1) = 3;
beta(2,1) = -2;
beta(4,1) = 1.5;
% beta(randperm(d,d/10*8)) = beta(randperm(d,d/10*8))*0;


z = mvnrnd(mu,var, n);
y = z*beta + sigma*randn(n,nr);
csvwrite('synthetic_data.csv',z)
csvwrite('synthetic_label.csv',y)
csvwrite('True_Theta.csv',beta')


% z = mvnrnd(mu,var, n);
% y = z*beta + sigma*randn(n,1);
% data = [z y];
% csvwrite('synthetic_data_val.csv',z)
% csvwrite('synthetic_label_val.csv',y)

n = 1000;
z = mvnrnd(mu,var, n);
y = z*beta + sigma*randn(n,nr);
csvwrite('synthetic_data_test.csv',z)
csvwrite('synthetic_label_test.csv',y)
