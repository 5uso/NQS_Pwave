from torch.autograd import grad
import itertools as it
from tqdm import tqdm
from torch import nn
import numpy as np
import torch

def roundrobin(*iterables):
    "Visit input iterables in a cycle until each is exhausted."
    # Recipe credited to George Sakkis
    n, nexts = len(iterables), it.cycle(iter(it).__next__ for it in iterables)
    while n:
        try:
            for next in nexts: yield next()
        except StopIteration: nexts = it.cycle(it.islice(nexts, n := n - 1))

class NQS(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, weights = None, biases = None, activation_fun = None):
        super(NQS, self).__init__()

        self.n_in, self.n_out, self.n_hidden = n_in, n_out, n_hidden

        self.weights = [torch.rand(n_layer, n_prev, requires_grad=True) for n_layer, n_prev in zip(it.chain(n_hidden, [n_out]), it.chain([n_in], n_hidden))] if weights is None else weights
        self.biases = [torch.rand(n_layer, requires_grad=True) for n_layer in n_hidden] if biases is None else biases

        self.activation_fun = nn.Sigmoid() if activation_fun is None else activation_fun
        self.linear_ops = nn.ModuleList([nn.Linear(in_layer, out_layer, True) for in_layer, out_layer in zip(it.chain([n_in], n_hidden[:-1]), n_hidden)] + [nn.Linear(n_hidden[-1], n_out, False)])

        with torch.no_grad():
            for layer, weight, bias in zip(self.linear_ops, self.weights, self.biases):
                layer.weight = nn.Parameter(weight)
                layer.bias = nn.Parameter(bias)
            self.linear_ops[-1].weight = nn.Parameter(self.weights[-1])

    def forward(self, x):
        for op in roundrobin(self.linear_ops, it.repeat(self.activation_fun, len(self.linear_ops) - 1)):
            x = op(x)
        return x

device = torch.device('cuda')

n_fermions = 4
v, sigma = -13.262, 0.375
gaussian_factor = v / np.sqrt(2 * np.pi) / sigma
n_hidden = [32, 32, 32]

epochs = 2000
learning_rate = 1e-2

n_mesh = 10
train_bounds = (-2.5, 2.5)
train_space = torch.linspace(train_bounds[0], train_bounds[1], n_mesh, requires_grad=True, device=device)
train_mesh = torch.cartesian_prod(*((train_space,) * n_fermions)).reshape(*((n_mesh,) * n_fermions), n_fermions)
fermions_x = [train_mesh[..., x] for x in range(n_fermions)]
train_mesh = torch.stack(fermions_x, dim=-1)
train_mesh_det = train_mesh.clone().detach()

h = (train_space[1].item() - train_space[0].item())**n_fermions
integration_weights = torch.full((n_mesh,) * n_fermions, h, device=device)

neural_state = NQS(n_fermions, 1, n_hidden).to(device)
optimizer = torch.optim.Adam(params=neural_state.parameters(), lr=learning_rate)

def loss_fn():
    psi = neural_state(train_mesh).squeeze()
    psi2 = neural_state(train_mesh_det).pow(2).squeeze()
    n = torch.tensordot(psi2, integration_weights, dims=n_fermions)

    potential = torch.tensordot(train_mesh_det.pow(2).sum(-1) * psi2, integration_weights, dims=n_fermions) / n / 2

    kinetic = torch.tensordot(sum(grad(outputs=psi, inputs=x, grad_outputs=torch.ones_like(psi), create_graph=True)[0].pow(2) for x in fermions_x), integration_weights, dims=n_fermions) / n / 2

    interaction = gaussian_factor * torch.tensordot(torch.exp(-sum((xi - xj).pow(2) for xi, xj in it.combinations(fermions_x, 2)) / (2*sigma**2)) * psi2, integration_weights, dims=n_fermions) / n / 2

    symmetry = 10 * torch.tensordot(sum((psi + neural_state(train_mesh[..., permutation]).squeeze()).pow(2) for permutation in [list(range(0, a)) + [b] + list(range(a+1, b)) + [a] + list(range(b+1, n_fermions)) for a, b in it.combinations(range(n_fermions), 2)]), integration_weights, dims=n_fermions) / n

    return potential + kinetic + interaction + symmetry, psi / torch.sqrt(n), potential + kinetic + interaction, potential, kinetic, interaction

losses, potentials, kinetics, interactions = [], [], [], []
energy = np.inf
for i in tqdm(range(epochs), desc="Training the NQS..."):
    loss, psi, e, pot, kin, int = loss_fn()

    losses.append(loss.item())
    potentials.append(pot.item())
    kinetics.append(kin.item())
    interactions.append(int.item())
    energy = min(energy, e)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'V={v}, sigma={sigma:.2f}, Energy: {energy.item():.4f} (V: {pot.item():.4f}, T: {kin.item():.4f}, I: {int.item():.4f})')
