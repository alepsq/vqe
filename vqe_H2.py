import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt
from functions import hydrogen_hamiltonian, hf, non_zero_state_amplitudes, plot

bohr_angs = 0.529177210903

def run_VQE(coordinates, output="all"):
    
    hamiltonian = hydrogen_hamiltonian(np.array(coordinates))
    num_qubits = len(hamiltonian.wires)
    hf_state = hf(num_qubits)
    singles, doubles = qml.qchem.excitations(2, num_qubits)

    dev = qml.device("default.qubit", wires=num_qubits)

    def ansatz(params):
        qml.AllSinglesDoubles(params, range(num_qubits), hf_state, singles, doubles)

    @qml.qnode(dev)
    def cost_fn(weights):
        qml.AllSinglesDoubles(weights, range(num_qubits), hf_state, singles, doubles)
        return qml.expval(hamiltonian)
    
    @qml.qnode(dev)
    def ground_state(params):
        ansatz(params)
        return qml.state()

    # Inizializzazione dei parametri
    np.random.seed(1234)
    theta_ = np.random.uniform(-np.pi, np.pi, len(singles) + len(doubles))
    theta = np.array(theta_, requires_grad=True)
    
    opt = qml.AdamOptimizer(0.03)
    energy = [cost_fn(theta)]
    max_iterations = 150

    for i in range(max_iterations):
        theta = opt.step(cost_fn, theta)
        energy.append(cost_fn(theta))
        
        if i > 0 and np.abs((energy[i] - energy[i-1]) / energy[i-1]) < 0.00001:
            break

    results = {
        "ground_state": ground_state(theta),
        "final_energy": energy[-1],
        "optimized_parameters": theta.tolist()
    }
    
    if output == "ground_state":
        return results["ground_state"]
    elif output == "final_energy":
        return results["final_energy"]
    elif output == "optimized_parameters":
        return results["optimized_parameters"]
    else:
        return results

def qa_ray(r):
    ground_state = []
    energy = []
    parameters = []
    
    for i in range(len(r)):
        
        
        coordinates = np.array([[0., 0., -r[i]/2], [0., 0., r[i]/2]])
        
        result = run_VQE(coordinates)
        
        final_energy = result["final_energy"]
        optimized_parameters = result["optimized_parameters"]
        gs = result["ground_state"]
        
        # Stampa formattata dei risultati per ogni r
        print(f"--- Distanza (r) = {r[i] * bohr_angs} Å ---")
        print(f"  Ground State: {gs}")
        print(f"  Energia finale: {final_energy:.6f} Ha")
        print(f"  Parametri ottimizzati: {', '.join([f'{param:.4f}' for param in optimized_parameters])}")
        print()
        
        
        # Memorizza i risultati nelle liste
        ground_state.append(gs)
        energy.append(final_energy)
        parameters.append(optimized_parameters)

        results ={
            "ground_state": ground_state,
            "energy": energy,
            "parameters": parameters
        }
        
    return results

def amplitudes(states):
    for i in range(len(states)):
        non_zero_amplitudes = non_zero_state_amplitudes(states[i])
        array_of_non_zero_amplitudes = []
        array_of_non_zero_amplitudes.append(non_zero_amplitudes)
    return array_of_non_zero_amplitudes



r_angstrom = np.array([0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.63, 0.66, 0.69, 0.70, 0.71, 0.715, 0.720, 0.725, 0.730, 0.731, 0.732, 0.733, 0.734, 0.735, 0.736, 0.737, 0.738, 0.739, 0.740, 0.742, 0.748, 0.75, 0.76, 0.78, 0.8, 0.85, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.])
r = r_angstrom / bohr_angs
output = qa_ray(r)
plot(r_angstrom, output["energy"])
#array_of_non_zero_amplitudes = amplitudes(output["ground_state"])

