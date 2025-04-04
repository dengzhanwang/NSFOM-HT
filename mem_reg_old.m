clear
clc
close all
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


Ai = A(1:batchsize,:);

bi = b(1:batchsize);
seed = 2024;
rng(seed);
%x_test = zeros(n,1); error = zeros(n,1); error(7) = 1e-5;

% f(x_test,Ai,bi);
%
%
% (f(x_test+error,Ai,bi) - f(x_test,Ai,bi) )/1e-5
%
% Df(x_test,Ai,bi)

x0 = zeros(n,1);

qmax = 5;

fv = zeros(epochmax+1,qmax+1);
mv = zeros(epochmax+1,qmax+1);
normdiff = [];
for q = -1:qmax


    p = q+1;

    zcs = zeros(n,q);

    Gzs = zeros(n,q);

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
        % if k == 1
        %     normdiff = [];
        %     for jj = 1:kmax
        %         idex2 = mod(jj-1,mdata);
        %         Ak2 = A(idex2+1:idex2+batchsize,:);
        %         bk2 = b(idex2+1:idex2+batchsize);
        %         normdiff = [normdiff, norm(Df(xc,Ak2,bk2) - Df(xc,A,b)) ];
        %     end
        % end

        if q == -1
            eta = 1/(k+1)^((2*.1+1)/(3*.1+1));
            gma = 1/(k+1)^((2*.1)/(3*.1+1));
            mc = (1-gma) * mm + gma * Df(xc,Ak,bk) + (1-gma) * (Df(xc,Ak,bk) - Df(xm,Ak,bk));
        elseif q == 4
            eta = 1/(k+1)^((2*1+1)/(3*1+1))/3;
            gma = 1/(k+1)^((2*1)/(3*1+1));
            mc = (1-gma) * mm + gma * Df(xc,Ak,bk);
        elseif q ==5
            eta = 1/(k+1)^((2*1+1)/(3*1+1))/2;
            gma = 1/(k+1)^((2*1)/(3*1+1));
            mc = (1-gma) * mm + gma * Df(xc,Ak,bk);
        elseif q == 0
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
            if q == 0
                xp = xc - eta * mc/norm(mc)^0.5/2.8;
            elseif q == 1
                xp = xc - eta * mc/norm(mc)^0.4/3.5;
            else
                xp = xc - eta * mc/norm(mc)^0.5/2;
            end
        elseif q == 4
            xp = xc - eta * mc/3;
        elseif q ==5
            xp = xc - eta * mc*min(2/norm(mc),1)/3;
        end
        % xp = xc - eta * mc/norm(mc);

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
% figure(1);
% set(gcf, 'Position', [100, 100, 800, 400]); % 设置窗口大小为 800x400 像素
% 
% % 第一个子图：直方图
% subplot(1, 2, 1);
% histogram(normdiff, 50); % 使用 30 个 bin 绘制直方图
% xlabel('Value');
% ylabel('Frequency');
% title('Histogram of normdiff');
% axis square; % 设置子图为正方形
% 
% % 第二个子图：QQ 图
% subplot(1, 2, 2);
% qqplot(normdiff);
% xlabel('Theoretical Quantiles');
% ylabel('Sample Quantiles');
% title('QQ Plot of normdiff');
% axis square; % 设置子图为正方形

figure(2)
set(gcf, 'Position', [100, 100, 1000, 450]);
title('red wine quality',FontSize=20)
subplot(1, 2, 1);
set(gca, 'FontWeight', 'bold', 'FontSize', 15)
fmin = min(fv,[],'all') - 5e-8;
semilogy((fv(:,2)-fmin)/(fv(1,2) - fmin ),'LineWidth',1.5);
hold on
semilogy((fv(1:0.5:epochmax/2,3)-fmin)/(fv(1,2) - fmin ),'LineWidth',1.5);
% plot(fv(:,4),'LineWidth',1.5);
% plot(fv(:,5),'LineWidth',1.5);
semilogy((fv(1:0.5:epochmax/2,1)-fmin)/(fv(1,2) - fmin ),'LineWidth',1.5);
semilogy((fv(:,6)-fmin)/(fv(1,2) - fmin ),'LineWidth',1.5);
semilogy((fv(:,7)-fmin)/(fv(1,2) - fmin ),'LineWidth',1.5);
ylim([1e-7, 1]);
set(gca, 'FontSize', 15)
% legend('NSG-PM','Algorithm 1 (q=1)','Algorithm 1 (q=2)','Algorithm 1 (q=3)','STORM','SG-PM','SG-PM-Clip' ,FontSize=11)
legend('NSG-PM','NSG-EXTRA','NSG-STORM','SG-PM','SG-PM-Clip',FontSize=11)
xlabel('gradient evalution number',FontSize=20)
ylabel('$\frac{f^k - f^*}{f^0 -f^*}$', 'Interpreter', 'latex', 'FontSize', 20);
% title('red wine quality',FontSize=20)

subplot(1, 2, 2);

semilogy(mv(:,2)/mv(1,2),'LineWidth',1.5);
hold on
semilogy(mv(:,3)/mv(1,2),'LineWidth',1.5);
% semilogy(mv(:,4),'LineWidth',1.5);
% semilogy(mv(:,5),'LineWidth',1.5);
semilogy(mv(:,1)/mv(1,2),'LineWidth',1.5);
semilogy(mv(:,6)/mv(1,2),'LineWidth',1.5);
semilogy(mv(:,7)/mv(1,2),'LineWidth',1.5);
ylim([1e-6, 1]);
set(gca, 'FontSize', 15)
%plot(fv(:,6));
legend('NSG-PM','NSG-EXTRA','NSG-STORM','SG-PM','SG-PM-Clip',FontSize=11)
xlabel('gradient evalution number',FontSize=20)
ylabel('$\frac{\|\nabla f(x^k) \|}{\|\nabla f(x^0) \|}$', 'Interpreter', 'latex', 'FontSize', 20);
saveas(gcf,'./result/mem_reg1.png')
1;
% figure(2)
% plot(fv(:,2),'LineWidth',1.5);
% hold on
% plot(fv(:,3),'LineWidth',1.5);
% plot(fv(:,4),'LineWidth',1.5);
% plot(fv(:,5),'LineWidth',1.5);
% plot(fv(:,1),'LineWidth',1.5);
% %plot(fv(:,6));
% legend('SG-PM','Algorithm 1 (q=1)','Algorithm 1 (q=2)','Algorithm 1 (q=3)','STORM',FontSize=11)
% xlabel('epoch',FontSize=14)
% ylabel('relative loss',FontSize=14)
% title('red wine quality',FontSize=15)
% 
% figure(3)
% plot(mv(:,2),'LineWidth',1.5);
% hold on
% plot(mv(:,3),'LineWidth',1.5);
% plot(mv(:,4),'LineWidth',1.5);
% plot(mv(:,5),'LineWidth',1.5);
% plot(mv(:,1),'LineWidth',1.5);
% %plot(fv(:,6));
% legend('SG-PM','Algorithm 1 (q=1)','Algorithm 1 (q=2)','Algorithm 1 (q=3)','STORM',FontSize=11)
% xlabel('epoch',FontSize=14)
% ylabel('gradient norm',FontSize=14)
% title('red wine quality',FontSize=15)


data_to_save = struct();
data_to_save.normdiff = normdiff;
data_to_save.fv = fv;
data_to_save.mv = mv;

save('wine_quality_data.mat', '-struct', 'data_to_save');
disp('数据已保存至 wine_quality_data.mat');

% figure
%
% semilogy(mv(:,2));
%
% hold on
%
% semilogy(mv(:,3));
%
% semilogy(mv(:,4));
%
%
% semilogy(mv(:,5));
%
% semilogy(mv(:,1));
%
% %plot(fv(:,6));
% legend('SG-PM','Algorithm 1 (q=1)','Algorithm 1 (q=2)','Algorithm 1 (q=3)','STORM')