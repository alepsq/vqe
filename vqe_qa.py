import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt

def hydrogen_hamiltonian(coordinates):
    molecule = qchem.Molecule(["H", "H"], coordinates)
    return qchem.molecular_hamiltonian(molecule)[0]

def hf(num_qubits):
    return qchem.hf_state(2, num_qubits)


def run_VQE(coordinates):
    hamiltonian = hydrogen_hamiltonian(np.array(coordinates))

    num_qubits = len(hamiltonian.wires)

    hf_state = hf(num_qubits)
    singles, doubles = qml.qchem.excitations(2, num_qubits)

    dev = qml.device("default.qubit", wires=num_qubits)

    def ansatz(params):
        return qml.AllSinglesDoubles(params, range (num_qubits), hf_state, singles, doubles)

    @qml.qnode(dev)
    def cost_fn(weights):

        """A circuit with tunable parameters/weights that measures the expectation value of the hydrogen Hamiltonian."""

        qml.AllSinglesDoubles(weights, range(num_qubits), hf_state, singles, doubles)
        return qml.expval(hamiltonian)
    @qml.qnode(dev)
    def ground_state(params):
        ansatz(params)
        return qml.state()

    np.random.seed(1234)
    theta_ = np.random.uniform(-np.pi, np.pi, len(singles) + len(doubles))
    theta = np.array(theta_, requires_grad = True)
    opt = qml.AdamOptimizer(0.03)
    angles = [theta]
    energy = [cost_fn(theta)]
    max_iterations = 150

    for i in range(max_iterations):
        theta = opt.step(cost_fn, theta)
        angles.append(theta)
        energy.append(cost_fn(theta))
        #print(theta)
        print(cost_fn(theta), i)
        if i > 0:
            if (np.abs((energy[i] - energy[i-1]) / energy[i-1]) < 0.00001):
                break
    return [ground_state(theta), cost_fn(theta), theta]

dev = qml.device("default.qubit", 4)
@qml.qnode(dev)
def expval_hamiltonian(theta,coordinates):
    qml.BasisState([1, 1, 0, 0], range(4))
    qml.SingleExcitation(theta[0], [0,2])
    qml.SingleExcitation(theta[1], [1, 3])
    qml.DoubleExcitation(theta[2], [0, 1, 2, 3])
    ham = hydrogen_hamiltonian(coordinates)
    return qml.expval(ham)

def qa_ray():
    coordinates_ = np.array([[-0.6, 0., 0.], [0.6, 0., 0.]])
    rays = []
    energies = []
    theta = run_VQE(coordinates_)[2]
    max_iterations = 150
    for i in range(1, max_iterations):
        r = 0.1*i
        rays.append(r)
        coordinates = np.array([[-r/2, 0., 0.], [r/2, 0., 0.]])
        energy = expval_hamiltonian(theta, coordinates)
        energies.append(energy)
        print(energy, i)
        if i > 2:
            if (np.abs((energies[i-1] - energies[i-2]) / energies[i-2]) < 0.001):
                break
    gs_energy = min(energies)
    gs_r = rays[energies.index(gs_energy)]
    return rays, energies, gs_r, gs_energy

def plot(r, energy):
    plt.scatter(r, energy, color="blue", label="Probabilità di |1>")
    plt.xlabel("r (Armstrong)")
    plt.ylabel("E (eV)")
    plt.title("H2 ground state")
    plt.legend()
    plt.show()
        
print(qa_ray())
ray = qa_ray()[0]
energy = qa_ray()[1]
plot(ray, energy)

    