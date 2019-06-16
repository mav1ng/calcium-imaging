import torch
import corr as c

a = torch.randint(0, 10, (5, 5), dtype=torch.double)

print(a)

a_ = torch.mean(a, dim=0)
a_n = a - a_
a_n_ = torch.sqrt(torch.sum(a_n ** 2, dim=0))
af = a_n / a_n_

b = a.clone().t()
b_ = torch.mean(b, dim=0)
b_n = b - b_
b_n_ = torch.sqrt(torch.sum(b_n ** 2, dim=0))
bf = (b_n / b_n_).t()

c = a.clone()
c_ = torch.mean(c, dim=0)
c_n = a - a_
c_n_ = torch.sqrt(torch.sum(c_n ** 2, dim=0))
cf = (c_n / c_n_)
cf_ = torch.mean(cf.t(), dim=0)
cf_n = cf.t() - cf_
cf_n_ = torch.sqrt(torch.sum(cf_n ** 2, dim = 0))
cff = (cf_n / cf_n_).t()
print(af)

print(b)
print(bf)

print(c)
print(cf)
print(cff)