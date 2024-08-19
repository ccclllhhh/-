x = [0.1 0.4 0.3 0.3 0.2];
y = qf_clr(x)
xx = qf_inverse_clr(y)


function y =qf_clr(x)
    % CLR 中心对数比变化
    fm = geomean(x); %几何平均
    y = log(x/fm)
end

function xx = qf_inverse_clr(y)
    % CLR 中心对数比变化的逆变换
    xx = exp(y)
    xx = xx/sum(xx);
end