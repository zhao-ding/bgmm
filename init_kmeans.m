function par = init_kmeans(x, K, rep)
% run k-means algorithm 'rep' times
% 
%   x   N x d   feature vector
%   K           number of components
%   rep         number of repetition
% 
if nargin<3,	rep=1;      end

regVal = eps(max(eig(cov(x))));

par = cell(rep,1);
[N,d] = size(x);

for tt=1:rep
    
    [idx, cent] = kmeans(x, K);
    
    pp = zeros(1,K);
    mu = zeros(K,d);
    sig = zeros(d,d,K);
    
    for k=1:K
        pp(k) = sum(idx==k);
        mu(k,:) = cent(k,:);
        if pp(k)
            sig(:,:,k) = cov(x(idx==k,:));
        end
        sig(:,:,k) = sig(:,:,k) + regVal*eye(d);
    end
    
    param.pp = pp / N;
    param.mu = mu;
    param.C = sig;
    
    par{tt} = param;
end

end
