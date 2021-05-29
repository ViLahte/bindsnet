import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_performance,
    plot_assignments,
    plot_voltages,
)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=1)
parser.add_argument("--update_interval", type=int, default=256)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true", default=False)
parser.add_argument("--gpu", dest="gpu", action="store_true", default=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
batch_size = args.batch_size
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu


if n_train % batch_size != 0:
    n_train = n_train - n_train % batch_size
    print(f'Warning: n_train should be a multiple of batch_size, using the closest value {n_train}.')

if n_test % batch_size != 0:
    n_test = n_test - n_test % batch_size
    print(f'Warning: n_test should be a multiple of batch_size, using the closest value {n_test}.')

if update_interval % batch_size != 0:
    update_interval = update_interval - update_interval % batch_size
    print(f'Warning: update_interval should be a multiple of batch_size, using value {update_interval}.')

assert update_interval % batch_size == 0 and update_interval >= batch_size, 'update_interval has to be a multiple of batch_size'
assert n_train % batch_size == 0 and n_train >= batch_size, 'n_train has to be a multiple of batch_size'
assert n_test % batch_size == 0 and n_test >= batch_size, 'n_test has to be a multiple of batch_size'

# # Enables interactive plotting.
# if plot:
#     mpl.use('TkAgg')

# Uses GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use.
if n_workers == -1:
    n_workers = 0  # gpu * 1 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Build network.
network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    nu=(1e-4, 1e-2),
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

# Copies the network into the GPU.
if gpu:
    network.to("cuda")

# Loads MNIST data for test and training sets.
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join(ROOT_DIR, "data", "MNIST"),
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join(ROOT_DIR, "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Ai"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

# Sets up tensor for holding spikes during batches.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Train the network.
print("Begin training.")
start = t()
for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("\nProgress (epoch): %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu
    )

    pbar_training = tqdm(total=n_train, position=0)
    pbar_training.set_description_str("Train progress: ")  
    for i, batch in enumerate(train_dataloader):
        if (i * batch_size) > n_train:
            break
        # Get next input sample.
        if gpu:
            inputs = {"X": batch["encoded_image"].to("cuda")}
        else:
            inputs = {"X": batch["encoded_image"]}

        if (i * batch_size) % update_interval == 0 and i > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f (last sample), %.2f (average over batches), %.2f (best over batches)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last sample), %.2f (average over batches), %.2f"
                " (best over batches)"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        labels.extend(batch["label"].tolist())

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Record spikes.
        s = spikes["Ae"].get("s").permute((1, 0, 2))
        start_index = (i * batch_size % update_interval)
        spike_record[start_index : start_index + s.size(0)] = s

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # Optionally plot simulation information at update intervals
        if plot and (i * batch_size) % update_interval == 0 and i > 0:
            image = batch["image"][:, 0].view(28, 28)
            input = inputs["X"][:, 0].view(time, 784).sum(0).view(28, 28)
            label = batch["label"][0]

            input_exc_weights = network.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {
                layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
            }
            voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, input, label=label, axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )

            plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.
        pbar_training.update(batch_size)

print("\nProgress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")



# Create a DataLoader to iterate and batch data.
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_workers,
    pin_memory=gpu
)


# Test the network.
print("Begin testing.")
network.train(mode=False)
start = t()
labels = []

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

pbar_testing = tqdm(total=n_test, position=0)
pbar_testing.set_description_str("Test progress: ")  
for i, batch in enumerate(test_dataloader):
    if (i * batch_size) > n_test:
        break
    # Get next input sample.
    if gpu:
        inputs = {"X": batch["encoded_image"].to("cuda")}
    else:
        inputs = {"X": batch["encoded_image"]}

    if (i * batch_size) % update_interval == 0 and i > 0:
        # Convert the array of labels into a tensor
        label_tensor = torch.tensor(labels, device=device)

        # Get network predictions.
        all_activity_pred = all_activity(
            spikes=spike_record, assignments=assignments, n_labels=n_classes
        )
        proportion_pred = proportion_weighting(
            spikes=spike_record,
            assignments=assignments,
            proportions=proportions,
            n_labels=n_classes,
        )

        # Compute network accuracy according to available classification strategies.
        accuracy["all"].append(
            100
            * torch.sum(label_tensor.long() == all_activity_pred).item()
            / len(label_tensor)
        )
        accuracy["proportion"].append(
            100
            * torch.sum(label_tensor.long() == proportion_pred).item()
            / len(label_tensor)
        )

        print(
            "\nAll activity accuracy: %.2f (last sample), %.2f (average over batches), %.2f (best over batches)"
            % (
                accuracy["all"][-1],
                np.mean(accuracy["all"]),
                np.max(accuracy["all"]),
            )
        )
        print(
            "Proportion weighting accuracy: %.2f (last sample), %.2f (average over batches), %.2f"
            " (best over batches)"
            % (
                accuracy["proportion"][-1],
                np.mean(accuracy["proportion"]),
                np.max(accuracy["proportion"]),
            )
        )

        # Assign labels to excitatory layer neurons.
        assignments, proportions, rates = assign_labels(
            spikes=spike_record,
            labels=label_tensor,
            n_labels=n_classes,
            rates=rates,
        )

        labels = []

    labels.extend(batch["label"].tolist())

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # Add spikes to recording.
    s = spikes["Ae"].get("s").permute((1, 0, 2))
    start_index = (i * batch_size % update_interval)
    spike_record[start_index : start_index + s.size(0)] = s

    network.reset_state_variables()  # Reset state variables.
    pbar_testing.update(batch_size)

print("\nTesting complete (%.4f seconds).\n" % (t() - start))
