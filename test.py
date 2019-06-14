import torch
import corr as c

a = torch.randint(0, 10, (5, 5), dtype=torch.float)

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

print(af)

print(b)
print(bf)


corrtest = torch.randint(0, 10, (3, 10, 10), dtype=torch.float)
print(c.get_corr(corrtest, corr_form='small_star', device=torch.device('cpu')))