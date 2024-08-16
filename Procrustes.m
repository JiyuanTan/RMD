rng(223)
dim_list = [5000];%[1000, 2000, 3000, 4000, 5000];
time_rec = zeros(2,length(dim_list));
obj_rec = zeros(2,length(dim_list));
for k = 1:length(dim_list)
dim = dim_list(k);
p = 10;
A = rand(dim)/sqrt(dim);
[X_opt,~] = qr(randn(dim,p),'econ');
B = A*X_opt;

% A_e = eigs(A,p,'largestreal');

% Set up problem
% prob.A = A;
% prob.p = p;
% prob.x0 = [eye(p);zeros(dim-p,p)];

[opt.x0,~] = qr(randn(dim,p),'econ');%[eye(p);zeros(dim-p,p)];
opt.gtol = 1e-5;
opt.xtol = 1e-7;
opt.ftol = 1e-9;
opt.iter = 2000;
opt.etals = 0.1;
opt.BB_step = 0;
rep = 1;

%fun.obj = @(X)norm((A*X-B))^2;
fun.obj = @(X)grad(X,A,B);

labels = [];

opt.div = 1;
opt.stepsize = 1e-3;
tic;
    res = md_bcd(opt,fun);
t = toc;
time_rec(1,k) = t;
obj_rec(1,k) = res.obj;
% semilogy(1:res.iter+1,-res.jh,LineWidth=1.2);
% hold on
% labels = sprintf("K = %d, time: %.2e",opt.div,t);

for div = [floor(dim/300)]
    totol_t = 0;
    jh = zeros(opt.iter+1,1);
    obj = 0;
    for r = 1:rep
        opt.stepsize = 1e-2;
        opt.div = div;
        opt.iter = 2000;

        tic;
        res_scgd = md_bcd(opt,fun);
        t = toc;
        %jh = jh + res_scgd.jh;
        jh = res_scgd.jh;
        obj = obj + res_scgd.obj;
        totol_t = totol_t + t;
        %     semilogy(1:(1/div^2):res.iter*(1/div^2)+1,sum(A_e) - res.jh,LineWidth=1.2);
    end
    totol_t = totol_t/rep;
    jh = jh/rep;
    time_rec(2,k) = totol_t/rep;
    obj_rec(2,k) = obj/rep;
%     semilogy(1:res_scgd.iter+1,jh,LineWidth=1.2);
%     hold on
%     labels = [labels,sprintf("K = %d, time: %.2e",div,totol_t)];
end
end

%% Run the algorithm
% res = stiefel_eig(prob,opt);

% res = md_bcd(opt,fun);

% check the result

% if norm(sum(A_e) - res.obj) < opt.tol
%     disp("Success!")   
% else
%     fprintf("Fail! The error is %e\n",norm(sum(A_e) - res.obj));
% end

%% Plot
% ylabel("Function Value");
% xlabel("Iteration");
% title(sprintf("Procrustes Problem with (n,p) = (%d,%d)",dim,p));
% legend(labels)

function [F,G] = grad(X,A,B)
    G = 2*A'*(A*X - B);
    F = norm((A*X-B))^2;
%     W = G*X'-X*G';
end
