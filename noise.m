% 生成满足 p(t) = 3/(4(1+|t|)^2.5) 的随机数
N = 1e6;                            % 样本数量
U1 = rand(N, 1);                     % 均匀分布 U[0,1)
V = (1./(1 - U1)).^(2/3) - 1;       % 生成绝对值部分 |t|
S = sign(rand(N, 1) - 0.5);          % 随机符号（±1）
t = S .* V;                          % 合并符号和绝对值

% 验证：绘制直方图与理论密度曲线
histogram(t, 'BinEdges', -10:0.2:10, 'Normalization', 'pdf');
hold on;
x = linspace(-10, 10, 1000);
pdf = 3./(4*(1 + abs(x)).^2.5);
plot(x, pdf, 'r', 'LineWidth', 2);
xlabel('t'); ylabel('p(t)');
legend('样本直方图', '理论密度');