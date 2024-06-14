import matplotlib.pyplot as plt
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

n_fermions = 2
v = -20
sigma = 0.5
gaussian_factor = v / np.sqrt(2 * np.pi) / sigma
n_hidden = [32, 16, 8]

epochs = 2000
learning_rate = 1e-2 #2e-3

n_mesh = 200
train_bounds = (-4, 4)
train_space = torch.linspace(train_bounds[0], train_bounds[1], n_mesh, requires_grad=True, device=device)
train_mesh = torch.cartesian_prod(*((train_space,) * n_fermions)).reshape(*((n_mesh,) * n_fermions), n_fermions)
fermions_x = [train_mesh[:, :, x] for x in range(n_fermions)]
train_mesh = torch.stack(fermions_x, dim=-1)
train_mesh_det = train_mesh.clone().detach()

h = (train_space[1].item() - train_space[0].item())**2
integration_weights = torch.full((n_mesh, n_mesh), h, device=device)

neural_state = NQS(n_fermions, 1, n_hidden).to(device)
#optimizer = torch.optim.RMSprop(params=neural_state.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(params=neural_state.parameters(), lr=learning_rate)

def loss_fn():
    psi = neural_state(train_mesh).squeeze()
    psi2 = neural_state(train_mesh_det).pow(2).squeeze()
    n = torch.tensordot(psi2, integration_weights)

    potential =  torch.tensordot(train_mesh_det.pow(2).sum(-1) * psi2, integration_weights) / n / 2

    dpsi_dx = [grad(outputs=psi, inputs=x, grad_outputs=torch.ones_like(psi), create_graph=True)[0] for x in fermions_x]
    dpsi2_dx2 = [d.pow(2) for d in dpsi_dx]
    kinetic = torch.tensordot(sum(dpsi2_dx2), integration_weights) / n / 2

    interaction = gaussian_factor * torch.tensordot(torch.exp(-(fermions_x[0] - fermions_x[1]).pow(2) / (2*sigma**2)) * psi2, integration_weights) / n / 2 # Gaussian interaction
    #directed2 = (dpsi_dx[0] - dpsi_dx[1]).pow(2)
    #interaction = gaussian_factor * torch.tensordot(torch.exp(-(fermions_x[0] - fermions_x[1]).pow(2) / (2*sigma**2)) * directed2, integration_weights) / n / 2 # Directed derivative gaussian interaction
    #interaction = torch.zeros(1, device=device) #No interaction

    # Weighing the symmetry constraint too heavily causes training to become excruciatingly slow
    psi_t = neural_state(train_mesh.flip(-1)).squeeze()
    #symmetry = 10 * torch.tensordot((psi2 - psi2_t).pow(2), integration_weights) / n # Symmetry: enforces psi(x1, x2) = psi(x2, x1)
    symmetry = 10 * torch.tensordot((psi + psi_t).pow(2), integration_weights) / n # Antisymmetry enforces psi(x1, x2) = -psi(x2, x1)

    return potential + kinetic + interaction + symmetry, psi / torch.sqrt(n), potential + kinetic + interaction, potential, kinetic, interaction

contour_levels = [-0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
density_dum = torch.rand((n_mesh, n_mesh))
x_plot = train_space.clone().detach().cpu().numpy()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
ax1, ax2, *_ = ax
fcont = ax1.contourf(x_plot, x_plot, density_dum, contour_levels, cmap='PRGn')
cont = ax1.contour(x_plot, x_plot, density_dum, contour_levels, colors='k')
ax1.axis("square")
bar = plt.colorbar(fcont, ticks=contour_levels)
ax1.set_xlabel("Position, $x_1$")
ax1.set_ylabel("Position, $x_2$")
ax1.set_xlim(train_bounds)
ax1.set_ylim(train_bounds)
ax2.plot([], label='Loss', color='tab:purple')
ax2.plot([], label='Potential')
ax2.plot([], label='Kinetic')
ax2.plot([], label='Interaction')
#ax2.set_yscale('symlog')
#ax2.yaxis.set_major_locator(LogLocator(base=2,numticks=6))
#ax2.yaxis.set_minor_locator(LogLocator(base=2,subs=[i / 4 for i in range(1, 4)],numticks=12))
#ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.legend()
fig.suptitle(f'$N={n_fermions}, V={v}, \sigma={sigma:.2f}$')
plt.ion()
plt.show()

def update_plot(wf, e, pot, kin, int):
    global fcont, cont, bar
    fcont.remove(); cont.remove()
    #while ax1.collections: ax1.collections.pop()
    fcont = ax1.contourf(x_plot, x_plot, wf, contour_levels, cmap='PRGn')
    cont = ax1.contour(x_plot, x_plot, wf, contour_levels, colors='k')

    x = np.linspace(1, len(e), len(e))
    ax2.set_xlim(0, len(e))
    ax2.set_ylim(min(a for a in it.chain(e, pot, kin, int) if a != 0), max(a for a in it.chain(e, pot, kin, int) if a != 0))
    for line, y in zip(ax2.lines, [e, pot, kin, int]):
        line.set_xdata(x)
        line.set_ydata(y)

    fig.canvas.draw()


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

    #if i % 200 == 0:
    #    update_plot(psi.clone().detach().cpu().numpy(), losses, potentials, kinetics, interactions)
    #plt.pause(0.01)

#print(losses)
update_plot(psi.clone().detach().cpu().numpy(), losses, potentials, kinetics, interactions)
print(f'V={v}, sigma={sigma:.2f}, Energy: {energy.item():.4f} (V: {pot.item():.4f}, T: {kin.item():.4f}, I: {int.item():.4f})')
input()
