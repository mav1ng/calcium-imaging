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

print(af)

print(b)
print(bf)

t = torch.randint(0, 10, (1, 10, 10, 10), dtype=torch.float)

def comp_similarity_matrix(input):
    """
    Method that computest the cosine similarity matrix
    input has dimensions Bs x Channels x Width x Height
    :param input:
    :return: N x N x 1 x Bs
    """
    (bs, ch, w, h) = input.size()
    out = torch.zeros((h * w, h * w, 1, bs))

    for i in range(bs):
        sim = input[i].view(ch, w * h)

        sim_ = torch.mean(sim, dim=0)
        sim_n = sim - sim_
        sim__ = torch.sqrt(torch.sum(sim_n ** 2, dim=0))
        sim = (sim_n / sim__).t()

        # sim_ = torch.mean(sim, dim=0)
        # sim_n = sim - sim_
        # sim__ = torch.sqrt(torch.sum(sim_n ** 2, dim=0))
        # sim = (sim_n / sim__)

        out[:, :, 0, i] = torch.mm(sim, sim.t()) * 0.5 + 0.5

    return out

res = comp_similarity_matrix(t)
print(res)
print(torch.sum(torch.where(res <= 1.0000, torch.tensor(0.), torch.tensor(1.))))
nonyero = (res > 1.).nonzero()
print(nonyero)