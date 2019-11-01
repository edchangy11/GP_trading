
clear all; close all;

rng(5)

% Need to add derivatives for both the Gram method and also the 


var_y = 1;

x = linspace(1,50,50)';
test_loc  = [25:29];
train_loc = [1:50];
N = length(x(train_loc));
l = 1;

% matern 3/2 Covariance matrix

K_s = cov_m(x(train_loc),x(train_loc),l);
K = K_s + var_y*eye(N);

mu = zeros(N,1);
y = mvnrnd(mu,K)';

Ntest = length(test_loc);
K_K_star = cov_m(x(train_loc),x(test_loc),l);
K_star_star = cov_m(x(test_loc),x(test_loc),l);


L = chol(K);
alpha = L'\(L\y);
f_star = K_K_star'*alpha;
v = L\K_K_star;
v_star = K_star_star - v'*v;
log_p_y = - 0.5*y'*alpha - sum(log(diag(L))) - (N/2).*log(2.*pi);

f_inv = K_K_star'*inv(K)*y;
cov_inv = K_star_star - K_K_star'*inv(K)*K_K_star;
log_p_2 = -0.5*y'*inv(K)*y - 0.5*log(det(K)) - N/2.*log(2.*pi);

%%

A_ = [0,1;-(sqrt(3)/l)^2,2*-sqrt(3)/l];
A = expm(A_);
H = [1,0];
P = [1,0;0,3/l^2];
Q = P - A*P*A';


y_kf = NaN*x;
y_kf(train_loc) = y;
[log_lik_ss,post_mean_ss_,post_cov_ss_,s,v] = kalmanSmoother(A,Q,H,P,var_y,y_kf);

fprintf('The log likelihood for state space is %d\n',log_lik_ss);
fprintf('The log likelihood for chol is %d\n',log_p_y);
fprintf('The log likelihood for inv is %d\n',log_p_2);

%% Computing the derivatives

param = [0.6,1];

K_u= cov_md(x(train_loc),x(train_loc),param);
alpha = inv(K)*y;
alpha_= alpha*alpha' - inv(K);
score = 0.5*trace(alpha_*K_u);


%t_s = 0.5*trace(inv(K)) - 0.5*alpha'*alpha
%var_y = var_y - 0.1*score;

log_n(K)
invK = L'\(L\eye(n)); 

invK = L'\(L\eye(N)); 
eg = 0.5*trace(invK) - 0.5*(alpha'*alpha);


%%

function K = cov_md(t1, t2, sigma2)
    dk = @(r,p) (1+sqrt(3)*abs(r)/p(2)).*exp(-sqrt(3)*abs(r)/p(2));
    K = zeros(length(t1), length(t2));
    for i=1:length(t1)
      for j=1:length(t2)
          r = abs(t1(i) - t2(j));
          K(i, j) =  dk(r,sigma2);        
          
      end
    end
end



function K = cov_m(t1, t2, l)
    K = zeros(length(t1), length(t2));
    for i=1:length(t1)
      for j=1:length(t2)
          r = abs(t1(i) - t2(j));
           K(i, j) =  (1 + sqrt(3)*r./l) .* exp((-sqrt(3)) * r ./ l);
      end
    end
end
  

  
  % Kalman filter and smoother for state space posterior calc
  function  [lik,Xfin,Pfin,s_,v_,a] = kalmanSmoother(A,Q,H,P,vary,y)
    T = length(y);
    s_ = nan(T,1);
    v_ = nan(T,1);
    a = nan(T,1);
    Y = reshape(y,[1,1,T]);
    lik=0;
    if length(vary) == 1
      vary = vary * ones(T,1);
    end
    m = zeros(size(A,1),1);
    MS = zeros(size(m,1),size(Y,3));
    PS = zeros(size(m,1),size(m,1),size(Y,3));
    Pfin = zeros(size(Y,3),sum(H));
    % ### Forward filter
    for k=1:T
        R = vary(k);
        % Prediction
        if (k>1)
            m = A*m;
            P = A*P*A' + Q;
        end
        % Kalman update
        if ~isnan(Y(:,:,k))
          S = H*P*H' + R; %innovation variance for likeli
          s_(k) = S;
          K = P*H'/S;
          v = Y(:,:,k)-H*m; % innovation mean for likeli
          v_(k) = v;
          a(k) = v/S;
          m = m + K*v;
          P = P - K*H*P; 
          % Evaluate the energy (neg. log lik): Check this
          lik = lik + .5*size(S,1)*log(2*pi) + .5*log(S) + .5*v^2/S;
        end
        PS(:,:,k) = P;
        MS(:,k)   = m;
    end
    % ### Backward smoother
    for k=size(MS,2)-1:-1:1
        % Smoothing step (using Cholesky for stability)
        PSk = PS(:,:,k);
        % Pseudo-prediction
        PSkp = A*PSk*A'+Q;
        [L,~] = chol(PSkp,'lower'); % Solve the Cholesky factorization
        % Continue smoothing step
        G = PSk*A'/L'/L;
        % Do update
        m = MS(:,k) + G*(m-A*MS(:,k));
        P = PSk + G*(P-PSkp)*G';
        MS(:,k)   = m;
        PS(:,:,k) = P;
        Pfin(k,:) = diag(P(find(H),find(H)));
    end
    lik = -lik;
    Xfin = reshape(MS,[1 size(MS)]);
    Xfin = squeeze(Xfin(1,find(H),:))';
  end
  