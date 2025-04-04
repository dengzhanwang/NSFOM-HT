clear all
clc
clf
T = readtable("winequality-red.csv");

raw_A = T{:,["fixedAcidity","volatileAcidity","citricAcid","residualSugar","freeSulfurDioxide","totalSulfurDioxide","density","pH","sulphates","alcohol"]};

raw_b = T.quality;
seed = 2024;
rng(seed)
% normalize the features

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


epochmax = 100;

kmax = mdata * epochmax;

%f = @(x,Ai,bi) sum((Ai*x-bi).^2./(1+(Ai*x-bi).^2))/(mdata * batchsize);

f = @(x) sum((x(1:end-1) - 1).^2) + (sum(x.^2) - 0.25)^2;

% 定义梯度的句柄函数
Df = @(x) [2 * (x(1:end-1) - 1); 0] + 4 * (sum(x.^2) - 0.25) * x...
            + 0*((1./(1 - rand(size(x)))).^(2/3) - 1).*sign(rand(size(x)) - 0.5);

% U1 = rand(N, 1);                     % 均匀分布 U[0,1)
% V = (1./(1 - U1)).^(2/3) - 1;       % 生成绝对值部分 |t|
% S = sign(rand(N, 1) - 0.5);          % 随机符号（±1）
% t = S .* V;         

%Df = @(x,Ai,bi) Ai' * (Ai*x - bi) + 0.01 * norm(x,4)^4;

%Df = @(x,Ai,bi) Ai'*(2*(Ai*x-bi)./(1+(Ai*x-bi).^2).^2)/(mdata * batchsize);


Ai = A(1:batchsize,:);

bi = b(1:batchsize);

%x_test = zeros(n,1); error = zeros(n,1); error(7) = 1e-5;

% f(x_test,Ai,bi);
%
%
% (f(x_test+error,Ai,bi) - f(x_test,Ai,bi) )/1e-5
%
% Df(x_test,Ai,bi)

x0 = ones(n,1);
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

    fvq(1) = f(xc);

    mvq(1) = norm(Df(xc));
    
    
    for k = 1:kmax
        idex = mod(k-1,mdata);
        Ak = A(idex+1:idex+batchsize,:);
        bk = b(idex+1:idex+batchsize);
        normdiff = [normdiff, norm(Df(xc) - Df(xc)) ];
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
            mc = (1-gma) * mm + gma * Df(xc) + (1-gma) * (Df(xc) - Df(xm));
        elseif q == 4
            eta = 1/(k+1)^((2*1+1)/(3*1+1))/3;
            gma = 1/(k+1)^((2*1)/(3*1+1));
            mc = (1-gma) * mm + gma * Df(xc);
        elseif q ==5
            eta = 1/(k+1)^((2*1+1)/(3*1+1))/3;
            gma = 1/(k+1)^((2*1)/(3*1+1));
            mc = (1-gma) * mm + gma * Df(xc);
        elseif q == 0
            eta = 1/(k+1)^((2*1+1)/(3*1+1))/3;
            gma = 1/(k+1)^((2*1)/(3*1+1));
            mc = (1-gma) * mm + gma * Df(xc);
        elseif q == 1
            eta = 1/(k+1)^((2*1.1+1)/(3*1.1+1))/2;
            gmas =  1/(k+1)^((2*1.1)/(3*1.1+1))*(1./((1:q)').^(1/10));
            V_extend = fliplr(vander([1;1./gmas]));
            V = V_extend(2:q+1,2:q+1);
            thas = (V')^(-1)*ones(q,1);
            for i = 1:q
                zcs(:,i) = xc + (1-gmas(i))*(xc - xm)/gmas(i);
                Gzs(:,i) = Df(zcs(:,i));
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
                Gzs(:,i) = Df(zcs(:,i));
            end
            mc = (1-sum(thas)) * mm +  Gzs * thas;
        end
        if q >= -1 && q<= 3
            if q == 0
            xp = xc - eta * mc/max(norm(mc)^0.7,1)/2.7;
            else
            xp = xc - eta * mc/norm(mc)^0.5;
            end
        elseif q == 4
            xp = xc - eta * mc/4.5;
        elseif q ==5
            xp = xc - eta * mc*min(10000/norm(mc),1)/4;
        end
        if mod(k,mdata) == 0
            fvq(k/mdata+1) = f(xp);
            mvq(k/mdata+1) = norm(Df(xp));
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
semilogy(fv(:,2)-min(fv,[],'all'),'LineWidth',1.5);
hold on
semilogy(fv(1:0.5:epochmax/2,3)-min(fv,[],'all'),'LineWidth',1.5);
% plot(fv(:,4),'LineWidth',1.5);
% plot(fv(:,5),'LineWidth',1.5);
semilogy(fv(1:0.5:epochmax/2,1)-min(fv,[],'all'),'LineWidth',1.5);
semilogy(fv(:,6)-min(fv,[],'all'),'LineWidth',1.5);
semilogy(fv(:,7)-min(fv,[],'all'),'LineWidth',1.5);
%plot(fv(:,6));
% legend('NSG-PM','Algorithm 1 (q=1)','Algorithm 1 (q=2)','Algorithm 1 (q=3)','STORM','SG-PM','SG-PM-Clip' ,FontSize=11)
legend('NSG-PM','NSG-EXTRA','NSG-STORM','SG-PM','SG-PM-Clip',FontSize=11)
xlabel('gradient evalution number',FontSize=14)
ylabel('relative loss',FontSize=14)
title('Extended Penalty function',FontSize=15)

figure(3)
semilogy(mv(:,2)/mv(1,2),'LineWidth',1.5);
hold on
semilogy(mv(1:0.5:epochmax/2,3)/mv(1,2),'LineWidth',1.5);
% semilogy(mv(:,4),'LineWidth',1.5);
% semilogy(mv(:,5),'LineWidth',1.5);
semilogy(mv(1:0.5:epochmax/2,1)/mv(1,2),'LineWidth',1.5);
semilogy(mv(:,6)/mv(1,2),'LineWidth',1.5);
semilogy(mv(:,7)/mv(1,2),'LineWidth',1.5);
%plot(fv(:,6));
legend('NSG-PM','NSG-EXTRA','NSG-STORM','SG-PM','SG-PM-Clip',FontSize=11)
xlabel('gradient evalution number',FontSize=14)
ylabel('gradient norm',FontSize=14)
title('Extended Penalty function',FontSize=15)


data_to_save = struct();
data_to_save.normdiff = normdiff;
data_to_save.fv = fv;
data_to_save.mv = mv;

save('noise2_data.mat', '-struct', 'data_to_save');
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