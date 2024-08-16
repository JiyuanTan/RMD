rng(223)
dim_list = [1000, 2000, 3000, 4000, 5000];
time_rec = zeros(2,length(dim_list));
obj_rec = zeros(2,length(dim_list));
for k = 1:length(dim_list)
dim = dim_list(k);
p = 10;
A = randn(dim); A = A'*A;
A_e = eigs(A,p,'largestreal');

% Set up problem
% prob.A = A;
% prob.p = p;
% prob.x0 = [eye(p);zeros(dim-p,p)];

opt.x0 = orth(randn(dim,p));%[eye(p);zeros(dim-p,p)];
opt.gtol = 1e-5;
opt.xtol = 1e-5;
opt.ftol = 1e-8;
opt.iter = 2000;
opt.etals = 0.1;
opt.ls_time = 5;

rep = 5;

% fun.ob = @(X)-dot(reshape(X,[dim*p,1]),reshape(A*X,[dim*p,1]))/2; 
fun.obj = @(X)grad(X,A);

labels = [];

opt.div = 1;
opt.stepsize = 1e-3;
opt.BB_step = 0;
tic;
    res = md_bcd(opt,fun);
t = toc;
time_rec(1,k) = t;
obj_rec(1,k) = res.obj/sum(A_e);
% semilogy(1:res.iter+1,sum(A_e) + 2*res.jh,LineWidth=1.2);
% hold on
% labels = sprintf("K = %d, time: %.1e,step:%.1e,error:%.3e",opt.div,t,opt.stepsize, (sum(A_e) + 2*res.obj)/sum(A_e));


opt.stepsize = 1e-2;
opt.BB_step = 0;
for div = [floor(dim/300)]
    totol_t = 0;
    %jh = zeros(opt.iter+1,1);
    obj = 0;
    for r = 1:rep
        opt.div = div;
        tic;
        res = md_bcd(opt,fun);
        t = toc;
        obj = obj + res.obj/sum(A_e);
        totol_t = totol_t + t;
        %     semilogy(1:(1/div^2):res.iter*(1/div^2)+1,sum(A_e) - res.jh,LineWidth=1.2);
    end
    time_rec(2,k) = totol_t/rep;
    obj_rec(2,k) = obj/rep;
    %semilogy(1:res.iter+1,sum(A_e) + 2*jh,LineWidth=1.2);
    %hold on
    %labels = [labels,sprintf("K = %d, time: %.1e,step:%.1e,error:%.3e",div,totol_t,opt.stepsize,(sum(A_e) + 2*res.obj)/sum(A_e))];
end
end
%% Plot
% ylabel("Optimality Gap");
% xlabel("Iteration");
% title(sprintf("n = %d,p = %d",dim,p));
% legend(labels)

function [F,G] = grad(X,A)
    G = - A*X;
    F = sum(dot(G,X,1))/2;
end
