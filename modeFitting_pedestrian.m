clear
clc
rng(123);
%% pedestrian
% for pedestrian : mixture_data:veh_velocity, ped_velocity, min_distance;
load('pedestrianEvents.mat')
% % distribution histo
% for i = 1:3
%    subplot(3,1,i)
%    hist(mixture_data(:,i),100)
% end
astar=[0 -7 5];
bstar=[13 6 60];


%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% best K = 6 %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
mixture_data=mixture_data(1:min(length(mixture_data),1000),:);
N_data=size(mixture_data,1);



%% centralize
% center=mean(mixture_data);
% var_Std=std(mixture_data);
% center_mat=repmat(center,N_data,1);
% var_Std_mat=repmat(var_Std,N_data,1);
%
% mixture_data=(mixture_data-center_mat)./var_Std_mat;
% astar=(astar-center)./var_Std;
% bstar=(bstar-center)./var_Std;



g_num_fun= @(data,mu_g,sigma_g) mvnpdf(data,mu_g,sigma_g)/((mvncdf([bstar(1),bstar(2),bstar(3)],mu_g,sigma_g)-mvncdf([astar(1),bstar(2),bstar(3)],mu_g,sigma_g))...
    +(mvncdf([bstar(1),astar(2),astar(3)],mu_g,sigma_g)-mvncdf([astar(1),astar(2),astar(3)],mu_g,sigma_g))...
    -(mvncdf([bstar(1),astar(2),bstar(3)],mu_g,sigma_g)-mvncdf([astar(1),astar(2),bstar(3)],mu_g,sigma_g))...
    -(mvncdf([bstar(1),bstar(2),astar(3)],mu_g,sigma_g)-mvncdf([astar(1),bstar(2),astar(3)],mu_g,sigma_g))...
    );

K_start = 3;
K_end = 15;
BIC_record = (K_start:K_end)*0;

for K=K_start:K_end  % Number of Mixtrue
    para_initial=init_kmeans(mixture_data,K);
    pi_hat=para_initial{1}.pp;
    mu_hat=para_initial{1}.mu;
    sigma_hat=para_initial{1}.C;
    
    eta_denom=0;
    eta_numer=zeros(1,K);
    
    
    
    for i=1:K
        eta_numer(i)=pi_hat(i)*((mvncdf([bstar(1),bstar(2),bstar(3)],mu_hat(i,:),sigma_hat(:,:,i))-mvncdf([astar(1),bstar(2),bstar(3)],mu_hat(i,:),sigma_hat(:,:,i)))...
            +(mvncdf([bstar(1),astar(2),astar(3)],mu_hat(i,:),sigma_hat(:,:,i))-mvncdf([astar(1),astar(2),astar(3)],mu_hat(i,:),sigma_hat(:,:,i)))...
            -(mvncdf([bstar(1),astar(2),bstar(3)],mu_hat(i,:),sigma_hat(:,:,i))-mvncdf([astar(1),astar(2),bstar(3)],mu_hat(i,:),sigma_hat(:,:,i)))...
            -(mvncdf([bstar(1),bstar(2),astar(3)],mu_hat(i,:),sigma_hat(:,:,i))-mvncdf([astar(1),bstar(2),astar(3)],mu_hat(i,:),sigma_hat(:,:,i)))...
            );
        eta_denom=eta_denom+eta_numer(i);
        
    end
    eta_hat=eta_numer/eta_denom;
    % Start EM algorithm
    converge_flag=false;
    N_count=0;
    N_max=50;% Number of max iterations
    
    eta_ite=zeros(1,K,N_max);
    mu_ite=zeros(K,3,N_max);
    sigma_ite=zeros(3,3,K,N_max);
    
    
    
    
    while ~converge_flag
        %     for maxit=1:300
        % E step
        z_n=zeros(N_data,K);
        z_n_nume=zeros(N_data,K);
        z_n_denom=zeros(N_data,1);
        for j=1:K
            z_n_nume(:,j)=eta_hat(j)*g_num_fun(mixture_data,mu_hat(j,:),sigma_hat(:,:,j));
            z_n_denom=z_n_denom+z_n_nume(:,j);
        end
        z_n=z_n_nume./repmat(z_n_denom,1,K);
        
        % M step
        eta_hat_new=mean(z_n);
        mu_hat_new=zeros(K,3);
        sigma_hat_new=zeros(3,3,K);
        
        % compute first and second moment
        
        
        for i=1:K
            [tmu,tcov,talpha]=tmvn_m3(zeros(1,3), sigma_hat(:,:,i), astar-mu_hat(i,:), bstar-mu_hat(i,:));
            m_k=tmu;
            H_k=sigma_hat(:,:,i)-(tcov+tmu'*tmu);
            
            mu_hat_new(i,:)=sum(repmat(z_n(:,i),1,3).*mixture_data)/sum(z_n(:,i))-m_k;
            
            sigma_add=zeros(3,3);
            for j=1:N_data
                sigma_add=sigma_add+z_n(j,i)*(mixture_data(j,:)-mu_hat_new(i,:))'*(mixture_data(j,:)-mu_hat_new(i,:));
            end
            sigma_hat_new(:,:,i)= sigma_add/sum(z_n(:,i))+H_k;
            
        end
        
        if all(abs(eta_hat-eta_hat_new)<0.01)&& all(all(abs(mu_hat-mu_hat_new)<0.01)) && all(all(all(abs(sigma_hat-sigma_hat_new)<0.01)))
            converge_flag=true;
        end
        
        N_count=N_count+1;
        
        
        if N_count>=N_max
            converge_flag=true;
        end
        disp(N_count)
        
        eta_ite(:,:,N_count)=eta_hat;
        mu_ite(:,:,N_count)=mu_hat;
        sigma_ite(:,:,:,N_count)=sigma_hat;
        
        eta_hat=eta_hat_new;
        mu_hat=mu_hat_new;
        sigma_hat=round(sigma_hat_new,6);
        
    end
    
    
    
    % Check BIC of distribution
    
    % compute log likelihood
    likelihood_value=zeros(N_data,1);
    for j=1:K
        part_likeli=eta_hat(j)*g_num_fun(mixture_data,mu_hat(j,:),sigma_hat(:,:,j));
        likelihood_value=likelihood_value+part_likeli;
    end
    log_likelihood=sum(log(likelihood_value));
    BIC=log(N_data)*K*(3+9+1)-2*log_likelihood;
    BIC_record(K - K_start + 1) = BIC;
    if K > K_start + 1
        if BIC_record(K - K_start + 1) > BIC_record(K- K_start)
            break;
        end
    end
    filename=['fitting_result_pedestrian_K = ' num2str(K)];
    save(filename,'BIC','eta_ite','mu_ite','sigma_ite')
end
BIC_figure = plot(K_start:K_end,BIC_record);
xlabel('K');
ylabel('BIC');
saveas(BIC_figure,'BIC_pedestrian.fig');