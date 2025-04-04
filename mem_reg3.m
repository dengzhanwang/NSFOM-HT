clear
clc
close all
addpath(genpath('./'));
T = readtable("winequality-red.csv");

raw_A = T{:,["fixedAcidity","volatileAcidity","citricAcid","residualSugar","freeSulfurDioxide","totalSulfurDioxide","density","pH","sulphates","alcohol"]};

raw_b = T.quality;

% normalize the features
raw_A = raw_A(1:1500,:);
raw_b = raw_b(1:1500,:);
[m,n] = size(raw_A);

A = zeros(m,n);

for i = 1:n
    amax = max(raw_A(:,i));
    amin = min(raw_A(:,i));
    A(:,i) = (raw_A(:,i) - amin)/(amax-amin);
end

bmax = max(raw_b);

bmin = min(raw_b);


b = (raw_b-bmin)/(bmax-bmin);

mdata = 5; batchsize = 100;


epochmax = 500;

kmax = mdata * epochmax;

%f = @(x,Ai,bi) sum((Ai*x-bi).^2./(1+(Ai*x-bi).^2))/(mdata * batchsize);

f = @(x,Ai,bi) norm(exp(Ai*x)./(1+exp(Ai*x))-bi,2)^2;

Df = @(x,Ai,bi) 2 * Ai' * ((exp(Ai*x)./(1+exp(Ai*x)).^2.*(exp(Ai*x)./(1+exp(Ai*x))-bi))) +  0.0001*((1./(1 - rand(size(x)))).^(2/3) - 1).*sign(rand(size(x)) - 0.5);

%Df = @(x,Ai,bi) Ai' * (Ai*x - bi) + 0.01 * norm(x,4)^4;

%Df = @(x,Ai,bi) Ai'*(2*(Ai*x-bi)./(1+(Ai*x-bi).^2).^2)/(mdata * batchsize);
seed = 2024;
rng(seed);
m = 2000;
n = 200;
A = randn(m,n);
x = rand(n,1);
b = exp(A*x)./(1+exp(A*x)) + 0.1*rand(m,1);


%x_test = zeros(n,1); error = zeros(n,1); error(7) = 1e-5;

% f(x_test,Ai,bi);
%
%
% (f(x_test+error,Ai,bi) - f(x_test,Ai,bi) )/1e-5
%
% Df(x_test,Ai,bi)

x0 = zeros(n,1);

qmax = 6;

fv = zeros(epochmax+1,qmax+1);
mv = zeros(epochmax+1,qmax+1);
normdiff = [];
for q = -1:qmax
    p = q+1;
    if q ~= 5
    zcs = zeros(n,q);

    Gzs = zeros(n,q);
    else
    zcs = zeros(n,1);

    Gzs = zeros(n,1);
    end

    eq = ones(q,1);

    fvq = zeros(epochmax+1,1);

    mvq = zeros(epochmax+1,1);

    xc = x0; xm = x0;

    mm = zeros(n,1);

    fvq(1) = f(xc,A,b);

    mvq(1) = norm(Df(xc,A,b));


    for k = 1:kmax
        idex = mod(k-1,mdata);
        % Ak = A(idex+1:idex+batchsize,:);
        % bk = b(idex+1:idex+batchsize);
        Ak = A;
        bk = b;
        normdiff = [normdiff, norm(Df(xc,Ak,bk) - Df(xc,A,b)) ];

        if q == -1
            eta = 1/(k+1)^((2*.1+1)/(3*.1+1));
            gma = 1/(k+1)^((2*.1)/(3*.1+1));
            mc = (1-gma) * mm + gma * Df(xc,Ak,bk) + (1-gma) * (Df(xc,Ak,bk) - Df(xm,Ak,bk));
        elseif q == 4
            eta = 1/(k+1)^((2*1+1)/(3*1+1))/3;
            gma = 1/(k+1)^((2*1)/(3*1+1));
            mc = (1-gma) * mm + gma * Df(xc,Ak,bk);
        elseif q ==5
            eta = 1/(k+1)^((2*1.1+1)/(3*1.1+1))/3;
            gmas =  1/(k+1)^((2*1.1)/(3*1.1+1))*(1./((1:1)').^(1/10));
            V_extend = fliplr(vander([1;1./gmas]));
            V = V_extend(2:1+1,2:1+1);
            thas = (V')^(-1)*ones(1,1);
            for i = 1:1
                zcs(:,i) = xc + (1-gmas(i))*(xc - xm)/gmas(i);
                Gzs(:,i) = Df(zcs(:,i),Ak,bk);
            end
            mc = (1-sum(thas)) * mm +  Gzs * thas;
        elseif q ==6
            eta = 1/(k+1)^((2*.1+1)/(3*.1+1))/3;
            gma = 1/(k+1)^((2*.1)/(3*.1+1));
            mc = (1-gma) * mm + gma * Df(xc,Ak,bk) + (1-gma) * (Df(xc,Ak,bk) - Df(xm,Ak,bk));
        elseif q == 0 % SGD
            eta = 1/(k+1)^((2*1+1)/(3*1+1))/3;
            gma = 1/(k+1)^((2*1)/(3*1+1));
            mc = (1-gma) * mm + gma * Df(xc,Ak,bk);

        elseif q == 1
            eta = 1/(k+1)^((2*1.1+1)/(3*1.1+1));
            gmas =  1/(k+1)^((2*1.1)/(3*1.1+1))*(1./((1:q)').^(1/10));
            V_extend = fliplr(vander([1;1./gmas]));
            V = V_extend(2:q+1,2:q+1);
            thas = (V')^(-1)*ones(q,1);
            for i = 1:q
                zcs(:,i) = xc + (1-gmas(i))*(xc - xm)/gmas(i);
                Gzs(:,i) = Df(zcs(:,i),Ak,bk);
            end
            mc = (1-sum(thas)) * mm +  Gzs * thas;
        else
            eta = 1/(k+1)^((2*1+1)/(3*1+1));
            gmas = 1/(k+1)^((2*1)/(3*1+1))*(1./((1:q)').^(1/10));
            V_extend = fliplr(vander([1;1./gmas]));
            V = V_extend(2:q+1,2:q+1);
            thas = (V')^(-1)*ones(q,1);
            for i = 1:q
                zcs(:,i) = xc + (1-gmas(i))*(xc - xm)/gmas(i);
                Gzs(:,i) = Df(zcs(:,i),Ak,bk);
            end
            mc = (1-sum(thas)) * mm +  Gzs * thas;
        end
        if q >= -1 && q<= 3
            if q == 0 % NSGD
                xp = xc - eta * mc/norm(mc)^0.5/1.9;
            elseif q == 1 % nEXTRA
                xp = xc - eta * mc/norm(mc)^0.5/2;
            else % nstorm
                xp = xc - eta * mc/norm(mc)^0.5/1.1;
            end
        elseif q == 4 % SGD
            xp = xc - eta * mc/20;
        elseif q ==5
            xp = xc - eta * mc/17;
        elseif q ==6
            xp = xc - eta * mc/10;
        end
        if mod(k,mdata) == 0
            fvq(k/mdata+1) = f(xp,A,b);
            mvq(k/mdata+1) = norm(Df(xp,A,b));
        end
        xm = xc; xc = xp; mm = mc;


    end

    fv(:,q+2) = fvq;

    mv(:,q+2) = mvq;
end

fv = fv/fv(1);
%mv = mv/mv(1);

%%
figure(2)
title('red wine quality',FontSize=20)

set(gca, 'FontWeight', 'bold', 'FontSize', 15)
fmin = min(fv,[],'all') - 1e-10;
semilogy((fv(:,2)-fmin)/(fv(1,2) - fmin), 'LineWidth', 2.5, 'LineStyle', '-', 'Color', 'r');
hold on
semilogy((fv(1:0.5:epochmax/2,3)-fmin)/(fv(1,2) - fmin), 'LineWidth', 2.5, 'LineStyle', '-', 'Color', 'b');
semilogy((fv(1:0.5:epochmax/2,1)-fmin)/(fv(1,2) - fmin), 'LineWidth', 2.5, 'LineStyle', '-', 'Color', 'm');

% 后三条虚线，颜色分别为红、蓝、黑
semilogy((fv(:,6)-fmin)/(fv(1,2) - fmin), 'LineWidth', 2.5, 'LineStyle', '--', 'Color', 'r');
semilogy((fv(:,7)-fmin)/(fv(1,2) - fmin), 'LineWidth', 2.5, 'LineStyle', '--', 'Color', 'b');
semilogy((fv(:,8)-fmin)/(fv(1,2) - fmin), 'LineWidth', 2.5, 'LineStyle', '--', 'Color', 'm');
ylim([1e-9, 10]);
set(gca, 'FontSize', 15)
% legend('NSG-PM','Algorithm 1 (q=1)','Algorithm 1 (q=2)','Algorithm 1 (q=3)','STORM','SG-PM','SG-PM-Clip' ,FontSize=11)
legend('NSFOM-PM','NSFOM-EM','NSFOM-RM','SFOM-PM','SFOM-EM','SFOM-RM',FontSize=11)
xlabel('gradient evalution',FontSize=20)
ylabel('relative objective value gap', 'Interpreter', 'latex', 'FontSize', 25);
saveas(gcf,'./result/mem_reg2_loss.png')
% title('red wine quality',FontSize=20)

figure(3)

semilogy(mv(:,2)/mv(1,2),'LineWidth',2.5,'LineStyle', '-', 'Color', 'r');
hold on
semilogy(mv(:,3)/mv(1,2),'LineWidth',2.5,'LineStyle', '-', 'Color', 'b');
% semilogy(mv(:,4),'LineWidth',2.5);
% semilogy(mv(:,5),'LineWidth',2.5);
semilogy(mv(:,1)/mv(1,2),'LineWidth',2.5,'LineStyle', '-', 'Color', 'm');
semilogy(mv(:,6)/mv(1,2),'LineWidth',2.5,'LineStyle', '--', 'Color', 'r');
semilogy(mv(:,7)/mv(1,2),'LineWidth',2.5,'LineStyle', '--', 'Color', 'b');
semilogy(mv(:,8)/mv(1,2),'LineWidth',2.5,'LineStyle', '--', 'Color', 'm');
ylim([1e-6, 1]);
set(gca, 'FontSize', 15)
%plot(fv(:,6));
legend('NSFOM-PM','NSFOM-EM','NSFOM-RM','SFOM-PM','SFOM-EM','SFOM-RM',FontSize=11)
xlabel('gradient evalution',FontSize=20)
ylabel('relative gradient norm', 'Interpreter', 'latex', 'FontSize', 25);
saveas(gcf,'./result/mem_reg2_gradient.png')
1;

data_to_save = struct();
% data_to_save.normdiff = normdiff;
% data_to_save.diffgrad = diffgrad;
data_to_save.fv = fv;
data_to_save.mv = mv;

save('mem_reg2_data.mat', '-struct', 'data_to_save');
disp('数据已保存至 mem_reg2_data.mat');