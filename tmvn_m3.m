
function [tmu, tcov, alpha] = tmvn_m3(mu, sig, astar, bstar)
% mean and covariance of a truncated normal distribution
% 

N = size(astar,1);
d = size(mu,2);

a = astar - mu;
b = bstar - mu;

% normalizing constant
alpha = mvncdf(a, b, zeros(1,d), sig);


% prevent numerical issue
an = a;
an(isinf(a)) = 0;
bn = b;
bn(isinf(b)) = 0;


% 1st moment
Fa = comp_marg1_new(a, sig, a, b);
Fb = comp_marg1_new(b, sig, a, b);

tEX = (Fa - Fb) * sig;
tEX = tEX ./ repmat(alpha, 1, d);

tmu = tEX + mu;


% 2nd moment
Faa = comp_marg2_new(a, a, sig, a, b);
Fab = comp_marg2_new(a, b, sig, a, b);
Fba = comp_marg2_new(b, a, sig, a, b);
Fbb = comp_marg2_new(b, b, sig, a, b);

F1 = an.*Fa - bn.*Fb;
F2 = Faa + Fbb - Fab - Fba;

tEXX = zeros(d,d,N);
tcov = zeros(d,d,N);
for n=1:N
    tEXX(:,:,n) = alpha(n) * sig + ...
        sig*diag(F1(n,:)./diag(sig)')*sig + ...
        sig*(F2(:,:,n) - diag(diag(F2(:,:,n)*sig)./diag(sig)))*sig;
    tEXX(:,:,n) = tEXX(:,:,n) / alpha(n);
    
    tcov(:,:,n) = tEXX(:,:,n) - tEX(n,:)'*tEX(n,:);
end

end


function Fk = comp_marg1_new(x, sig, a, b)

[N,d] = size(x);
Fk = zeros(N,d);

for k=1:d
    Fk(:,k) = tmvn_marg1_new(x(:,k), k, sig, a, b);
end

end


function Fkq = comp_marg2_new(xk, xq, sig, a, b)

[N,d] = size(xk);
Fkq = zeros(d,d,N);

for k=1:d
    for q=1:d
        if q==k
            continue;
        end
        
        Fkq(k,q,:) = tmvn_marg2_new(xk(:,k), xq(:,q), k, q, sig, a, b);
    end
end

end


function Fk = tmvn_marg1_new(x, k, sig, a, b)
% univariate marginal
% 

N = size(x,1);
Fk = zeros(N,1);

% if x = Inf or -Inf, then Fk = 0
idx = ~isinf(x);
if any(~idx)
    Fk(~idx) = 0;
end

if ~any(idx)
    return
end

% dimension
d = size(sig,1);

% univariate d=1
if d==1
    Fk(idx) = normpdf(x(idx), 0, sqrt(sig));
    return
end

% multivariate d>1
o = false(1,d);
o(k) = true;
m = ~o;

cmu = (sig(m,o) / sig(o,o) * x(idx)')';
csig = sig(m,m) - sig(m,o)/sig(o,o)*sig(o,m);

Fk(idx) = normpdf(x(idx), 0, sqrt(sig(o,o))) ...
    .* mvncdf(a(idx, m) - cmu, b(idx, m) - cmu, zeros(1,d-1), csig);

end


function Fkq = tmvn_marg2_new(xk, xq, k, q, sig, a, b)
% bivariate marginal
% 

if (k==q)
    warning('tmvn_marg2:dim', 'tmvn_marg2 : k should be different from q \n');
end

N = size(xk,1);
Fkq = zeros(N,1);

% if xk or xq = Inf or -Inf, then Fkq = 0
idx = ~(isinf(xk) | isinf(xq));
if any(~idx)
    Fkq(~idx) = 0;
end

if ~any(idx)
    return
end

% dimension
d = size(sig,1);

% univariate d=1
if d==1
    Fkq = 0;
    return
end

% change the order of variables
if k<q
    x = [xk, xq];
else
    x = [xq, xk];
end

% bivariate d=2
if d==2
    Fkq(idx) = mvnpdf(x(idx,:), [0 0], sig);
    return
end

% multivariate d>2
o = false(1,d);
o([k,q]) = true;
m = ~o;

cmu = (sig(m,o) / sig(o,o) * x(idx,:)')';
csig = sig(m,m) - sig(m,o) / sig(o,o) * sig(o,m);

Fkq(idx) = mvnpdf(x(idx,:), [0 0], sig(o,o)) ...
    .* mvncdf(a(idx,m) - cmu, b(idx, m) - cmu, zeros(1,d-2), csig);

end
