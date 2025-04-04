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

mdata = 5; batchsize = 300;


epochmax = 500;

kmax = mdata * epochmax;

%f = @(x,Ai,bi) sum((Ai*x-bi).^2./(1+(Ai*x-bi).^2))/(mdata * batchsize);


f = @(x, Ai, bi) sum((Ai * x - bi).^2 ./ (1 + (Ai * x - bi).^2));

% 梯度计算
Df = @(x, Ai, bi) 2 * Ai' * ((Ai * x - bi) ./ (1 + (Ai * x - bi).^2).^2);


Ai = A(1:batchsize,:);

bi = b(1:batchsize);
seed = 2024;
rng(seed);


x0 = zeros(n,1);

qmax = 6;
normdiff = [];
fv = zeros(epochmax+1,qmax+1);
mv = zeros(epochmax+1,qmax+1);
normdiff = [];
diffgrad = [];
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
        if k == 1
            batchsize2 = 5;
            mmm = size(A,1);
            iternum = floor(mmm/batchsize2);
            for jj = 0:iternum-1
                % idex2 = mod(jj-1,mdata);
                Ak2 = A(batchsize2*jj+1:batchsize2*(jj+1),:);
                bk2 = b(batchsize2*jj+1:batchsize2*(jj+1));
                normdiff = [normdiff, norm(Df(xc,Ak2,bk2) - Df(xc,A,b) +  200*((1./(1 - rand(size(xc)))).^(2/3) - 1).*sign(rand(size(xc)) - 0.5) )/1000 ];
            end
        end

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
            if norm(xc-xm) ~=0
                diffgrad = [diffgrad norm(Df(xc,Ak,bk) - Df(xm,Ak,bk))/norm(xc-xm) ];
            end
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
                xp = xc - eta * mc/norm(mc)^0.5/20;
            elseif q == 1 % nEXTRA
                xp = xc - eta * mc/norm(mc)^0.5/45;
            else % nstorm
                xp = xc - eta * mc/norm(mc)^0.5/12;
            end
        elseif q == 4 % SGD
            xp = xc - eta * mc/250;
        elseif q ==5
            xp = xc - eta * mc/150;
        elseif q ==6
            xp = xc - eta * mc/50;
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
% diffgrad = diffgrad(diffgrad<0.02);
fv = fv/fv(1);
% diffgrad = mvq(1:end-1) - mvq(2:end);
%mv = mv/mv(1);

%%
figure(1);
set(gcf, 'Position', [100, 100, 800, 400]); % 设置窗口大小为 800x400 像素

% 第一个子图：直方图
subplot(1, 2, 1);
histogram(normdiff, 50); % 使用 30 个 bin 绘制直方图
xlabel('Value');
ylabel('Frequency');
title('Histogram of normdiff');
axis square; % 设置子图为正方形

% 第二个子图：QQ 图
subplot(1, 2, 2);
qqplot(normdiff);
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
title('QQ Plot of normdiff');
axis square; % 设置子图为正方形



figure(2);
set(gcf, 'Position', [100, 100, 800, 400]); % 设置窗口大小为 800x400 像素

% 第一个子图：直方图
subplot(1, 2, 1);
histogram(diffgrad, 50); % 使用 30 个 bin 绘制直方图
xlabel('Value');
ylabel('Frequency');
title('Histogram of normdiff');
axis square; % 设置子图为正方形

% 第二个子图：QQ 图
subplot(1, 2, 2);
qqplot(diffgrad);
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
title('QQ Plot of normdiff');
axis square; % 设置子图为正方形

figure(3)
% title('red wine quality',FontSize=20)
% subplot(1, 2, 1);
set(gca, 'FontWeight', 'bold', 'FontSize', 15)
fmin = min(fv,[],'all') - 5e-8;
semilogy((fv(:,2)-fmin)/(fv(1,2) - fmin ),'LineWidth',2.5, 'LineStyle', '-', 'Color', 'r');
hold on
semilogy((fv(1:0.5:epochmax/2,3)-fmin)/(fv(1,2) - fmin ),'LineWidth',2.5, 'LineStyle', '-', 'Color', 'b');
% plot(fv(:,4),'LineWidth',2.5);
% plot(fv(:,5),'LineWidth',2.5);
semilogy((fv(1:0.5:epochmax/2,1)-fmin)/(fv(1,2) - fmin ),'LineWidth',2.5, 'LineStyle', '-', 'Color', 'm');
semilogy((fv(:,6)-fmin)/(fv(1,2) - fmin ),'LineWidth',2.5, 'LineStyle', '--', 'Color', 'r');
semilogy((fv(:,7)-fmin)/(fv(1,2) - fmin ),'LineWidth',2.5, 'LineStyle', '--', 'Color', 'b');
semilogy((fv(:,8)-fmin)/(fv(1,2) - fmin ),'LineWidth',2.5, 'LineStyle', '--', 'Color', 'm');
% ylim([5e-3, 1]);
set(gca, 'FontSize', 15)
% legend('NSG-PM','Algorithm 1 (q=1)','Algorithm 1 (q=2)','Algorithm 1 (q=3)','STORM','SG-PM','SG-PM-Clip' ,FontSize=11)
legend('NSG-PM','NSG-EXTRA','NSG-STORM','SG-PM','SG-EXTRA','SG-STORM',FontSize=11)
xlabel('gradient evalution number',FontSize=20)
ylabel('$\frac{f^k - f^*}{f^0 -f^*}$', 'Interpreter', 'latex', 'FontSize', 25);
saveas(gcf,'./result/mem_robust_reg1_loss.png')



figure(4)

semilogy(mv(:,2)/mv(1,2),'LineWidth',2.5, 'LineStyle', '-', 'Color', 'r');
hold on
semilogy(mv(:,3)/mv(1,2),'LineWidth',2.5, 'LineStyle', '-', 'Color', 'b');
% semilogy(mv(:,4),'LineWidth',2.5);
% semilogy(mv(:,5),'LineWidth',2.5);
semilogy(mv(:,1)/mv(1,2),'LineWidth',2.5, 'LineStyle', '-', 'Color', 'm');
semilogy(mv(:,6)/mv(1,2),'LineWidth',2.5, 'LineStyle', '--', 'Color', 'r');
semilogy(mv(:,7)/mv(1,2),'LineWidth',2.5, 'LineStyle', '--', 'Color', 'b');
semilogy(mv(:,8)/mv(1,2),'LineWidth',2.5, 'LineStyle', '--', 'Color', 'm');
% ylim([1e-1, 1]);
set(gca, 'FontSize', 15)
%plot(fv(:,6));
legend('NSG-PM','NSG-EXTRA','NSG-STORM','SG-PM','SG-EXTRA','SG-STORM',FontSize=11)
xlabel('gradient evalution number',FontSize=20)
ylabel('$\frac{\|\nabla f(x^k) \|}{\|\nabla f(x^0) \|}$', 'Interpreter', 'latex', 'FontSize', 25);
saveas(gcf,'./result/mem_robust_reg1_gradient.png')
1;
% figure(2)
% plot(fv(:,2),'LineWidth',2.5);
% hold on
% plot(fv(:,3),'LineWidth',2.5);
% plot(fv(:,4),'LineWidth',2.5);
% plot(fv(:,5),'LineWidth',2.5);
% plot(fv(:,1),'LineWidth',2.5);
% %plot(fv(:,6));
% legend('SG-PM','Algorithm 1 (q=1)','Algorithm 1 (q=2)','Algorithm 1 (q=3)','STORM',FontSize=11)
% xlabel('epoch',FontSize=14)
% ylabel('relative loss',FontSize=14)
% title('red wine quality',FontSize=15)
%
% figure(3)
% plot(mv(:,2),'LineWidth',2.5);
% hold on
% plot(mv(:,3),'LineWidth',2.5);
% plot(mv(:,4),'LineWidth',2.5);
% plot(mv(:,5),'LineWidth',2.5);
% plot(mv(:,1),'LineWidth',2.5);
% %plot(fv(:,6));
% legend('SG-PM','Algorithm 1 (q=1)','Algorithm 1 (q=2)','Algorithm 1 (q=3)','STORM',FontSize=11)
% xlabel('epoch',FontSize=14)
% ylabel('gradient norm',FontSize=14)
% title('red wine quality',FontSize=15)


data_to_save = struct();
data_to_save.normdiff = normdiff;
data_to_save.diffgrad = diffgrad;
data_to_save.fv = fv;
data_to_save.mv = mv;

save('redwine_quality_data.mat', '-struct', 'data_to_save');
disp('数据已保存至 redwine_quality_robust_regression_data.mat');
