% This program implement the randomized Stiefel Curvlinear descent
% algorithm
%　Input:
% (1) Option:
%           opt.step: stepsize
%           opt.iter: Maximum iteration
%           opt.div: divide the dimenension into opt.div parts
%           opt.x0: Initiate point
%           opt.gtol: tolerance
%           opt.rhols: line search decrease ratio
%           opt.etals: line search parameter
%           opt.gammals: line search parameter
% (2) Fun: fun.ob objective function
%          fun.grad gradient function
%               Note that fun.grad returns a n*n skew-symmetric matrix
%   Output:
% (1) result:
%       result.jh history of objective value
%       result.x minimizer

function result = md_bcd(opt,fun)
%% Initiate
if ~isfield(opt, 'rhols');     opt.rhols  = 1e-4; end
if ~isfield(opt, 'etals');     opt.etals  = 1e-1; end
if ~isfield(opt, 'gammals');     opt.gammals  = 0.85; end
if ~isfield(opt, 'xtol');      opt.xtol = 1e-6; end
if ~isfield(opt, 'ftol');      opt.ftol = 1e-12; end
if ~isfield(opt, 'gtol');      opt.gtol = 1e-4; end
if ~isfield(opt, 'BB_step');   opt.BB_step = 0; end
x = opt.x0;
Q = 1;
[n,p] = size(x);
eta = opt.stepsize;
result.jh = zeros(opt.iter+1,1);
if (norm(x'*x - eye(p))>1e-9)
    error("The initiate point is not orthgonal!")
end
iter = 1;
[j,G] = fun.obj(x);
result.neval = 1;
% W = G*x' - x*G';
g = G - x*(G'*x);
result.jh(iter) = j;
% choose line search base line value
Cval = result.jh(iter);
step = eta;
%% Main iteration
while iter < opt.iter + 1 && norm(g,"fro") > opt.gtol

    % Calculate the direction
    ind = randperm(n);
    info = update_info(x,G,opt.div,ind);
    x_can = cwise_update(x,info,step);
    [j,G_new] = fun.obj(x_can);
    result.neval = result.neval + 1;
    ls_time = 1;
    deriv = norm(g,"fro")^2; %derivation(g,ind,opt.div);
    %line search -- Hager-Zhang
    while j >= Cval - opt.rhols* step * deriv && ls_time < opt.ls_time
        step = step * opt.etals;
        x_can = cwise_update(x,info,step);
        [j,G_new] = fun.obj(x_can);
        result.neval = result.neval + 1;
        ls_time = ls_time + 1;
    end
    x_diff = x_can - x;
    x_change = norm(x_diff,"fro")/sqrt(n);
    x = x_can;
    %   Orthgonalization
    %   if (norm(x'*x - eye(p)) > opt.gtol)
    %      [x,~] = qr(x,"econ");
    %   end
    % Update line search parameters
    iter = iter + 1;
    result.jh(iter) = j;
    F_change = abs(j - result.jh(iter-1))/(1+abs(result.jh(iter-1)));
    G = G_new;
    % W = G*x' - x*G';

    g_new = G - x*(G'*x);
    if opt.BB_step == 1 
        g_diff = g_new - g;
        x_g = abs(sum(sum(g_diff.*x_diff)));
        if mod(iter, 2) == 1
            step = (norm(x_diff,'fro')^2)/x_g;
        else
            step =  x_g/(norm(g_diff,'fro')^2);
        end
        %step = step * opt.div;
    else
        step = eta;
    end
   
    g = g_new;
    % update line search parameters
    Qp = Q;
    Q = opt.gammals*Qp + 1;
    Cval = (opt.gammals * Qp * Cval + j)/Q;

    % Stop criterion
    if F_change <= opt.ftol || x_change <= opt.xtol
        break
    end
end
%% Output result
result.jh = result.jh(1:iter);
result.x = x;
result.obj = result.jh(end);
result.iter = iter - 1;
result.gnorm = norm(g,"fro");
end

function step = calculate_bbstep(g_diff,x_diff,iter,info)
% update step size
div = info.div;
ind = info.ind;
[n,~] = size(x_diff);
n_div = floor(n / div + 0.5);
step = ones(div,1);
for i = 1:div
    ind_block = ind((i-1)*n_div+1:min(i*n_div,n));
    x_g = abs(sum(sum(g_diff(ind_block).*x_diff(ind_block))));
    if mod(iter, 2) == 1
        step(i) = (norm(x_diff(ind_block),'fro')^2)/x_g;
    else
        step(i) =  x_g/(norm(g_diff(ind_block),'fro')^2);
    end
end
step = max(min(step, 1e12), 1e-12);
end

function info = update_info(x,G,div,ind)
[n,p] = size(x);
n_div = floor(n / div + 0.5);
info.div = div;
info.smw = 1;
info.ind = ind;
info.G = G;
if n_div <= 2 *p
    info.smw = 0;
else
    % use smw update
    info.VX = zeros(2*p, p * div);
    info.VU = zeros(2*p,2 * p * div);
    info.U = zeros(n_div,2*p*div);
    for i = 1:div
        ind_block = sort(ind((i-1)*n_div+1:min(i*n_div,n)));
        info.ind((i-1)*n_div+1:min(i*n_div,n)) = ind_block;
        U = [G(ind_block,:),x(ind_block,:)];
        V = [x(ind_block,:),-G(ind_block,:)];
        info.VX(1:2*p,(i-1)*p + 1:(i)*p) = V'*x(ind_block,:);
        info.VU(1:2*p,(i-1)*2*p + 1:(i)*2*p) = V'*U;
        info.U(1:length(ind_block),(i-1)*2*p + 1:(i)*2*p) = U;
    end
end
end

function x = cwise_update(x,info,eta)
div = info.div;
ind = info.ind;
[n,p] = size(x);
n_div = floor(n / div + 0.5);
for i = 1:div
    ind_block = ind((i-1)*n_div+1:min(i*n_div,n));
    if info.smw
        x(ind_block,:) = SMW_update(info.VX(1:2*p,(i-1)*p + 1:(i)*p),info.VU(1:2*p,(i-1)*2*p + 1:(i)*2*p),...
            info.U(1:length(ind_block),(i-1)*2*p + 1:(i)*2*p),x(ind_block,:),eta);
    else
        x(ind_block,:) = inverse_update(info.G(ind_block,:),x(ind_block,:),eta);
    end
end
end
function X = SMW_update(VX,VU,U,X,eta)
    [k,~] = size(VU);
    %temp = linsolve(eye(k) + (eta*0.5)*VU,VX);
    X = X - eta * (U * linsolve(eye(k) + (eta*0.5)*VU,VX));
end
function X = inverse_update(G,X,eta)
    [n,~] = size(X);
    W = G*X' - X* G';
    X = (eye(n) + eta/2 * W)\((eye(n) - eta/2 * W)*X);
end

% Calculate the derivation
function deriv = derivation(g,ind,div)
[n,~] = size(g);
% n_div = floor(n / div + 0.5);
% deriv = 0;
% for i = 1:div
%     ind_block = sort(ind((i-1)*n_div+1:min(i*n_div,n)));
%     deriv = deriv + norm(G(ind_block,ind_block),"fro")^2;
%     %       deriv = deriv + norm(W(ind_block,ind_block)*x(ind_block,:),"fro")^2;
% end
deriv = norm(g,"fro")^2;
end


%% Maybe you can use Cell to realize Parallel computing
function x = cwise_update_para(x,W,eta,div,ind)
[n,~] = size(x);
n_div = floor(n / div + 0.5);
x_para = cell(div,1);
for i = 1:div
    ind((i-1)*n_div+1:min(i*n_div,n)) = sort(ind((i-1)*n_div+1:min(i*n_div,n)));
    x_para{i,1} = x(ind((i-1)*n_div+1:min(i*n_div,n)),:);
end
parfor i = 1:div
    d_block = min(i*n_div,n) - (i-1)*n_div;
    ind_block = ind((i-1)*n_div+1:min(i*n_div,n));
    x_para{i,1} = (eye(d_block) + eta*W(ind_block,ind_block))*x_para{i,1};
    x_para{i,1} = (eye(d_block) - eta*W(ind_block,ind_block))\x_para{i,1};
end
for i = 1:div
    x(ind((i-1)*n_div+1:min(i*n_div,n)),:) = x_para{i,1};
end
end
