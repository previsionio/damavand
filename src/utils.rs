//! # utils
//!
//! `utils` is a collection of utilities to assist damavand.
//!

use crate::qubit_backend::gates;
use crate::qubit_backend::gates::Gate;
use crate::utils;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use num::complex::Complex;
use rand::prelude::*;
use std::sync::{Arc, Mutex};

#[derive(PartialEq, Debug)]
pub enum ApplyMethod {
    BruteForce,
    Shuffle,
    Multithreading,
    GPU,
    DistributedCPU,
    DistributedGPU,
}

///
///
///
///
///
///
///
pub fn kron<T>(a: &Array2<T>, b: &Array2<T>) -> Array2<T>
where
    T: LinalgScalar,
{
    let dima = a.shape()[0];
    let dimb = b.shape()[0];
    let dimout = dima * dimb;
    let mut out = Array2::zeros((dimout, dimout));
    for (mut chunk, elem) in out.exact_chunks_mut((dimb, dimb)).into_iter().zip(a.iter()) {
        let v: Array2<T> = Array2::from_elem((dimb, dimb), *(elem)) * b;
        chunk.assign(&v);
    }
    out
}

///
///
///
///
///
pub fn kron_shuffle(
    matrix_list: &Vec<Array2<Complex<f64>>>,
    state: &Array1<Complex<f64>>,
) -> Array1<Complex<f64>> {
    let mut q = state.clone();

    // initialize
    let num_matrices = matrix_list.len();

    // transpose matrices because the algorithm is written for left product
    let mut m_list = Vec::<Array2<Complex<f64>>>::new();

    for matrix_index in 0..num_matrices {
        m_list.push(matrix_list[matrix_index].clone().reversed_axes());
    }

    // get number of rows of each matrices
    let row_size: usize = 2;

    // initialize
    let mut ileft: usize = 1;
    let mut iright: usize = 1;
    for _ in &m_list[1..] {
        iright *= 2;
    }

    for h in 0..num_matrices {
        let mut base_i: usize = 0;
        let mut base_j: usize = 0;
        let mut z = Array::ones(row_size);
        for _ in 0..ileft {
            (0..iright).into_iter().for_each(|ir| {
                let mut ii: usize = base_i + ir;
                let mut ij: usize = base_j + ir;

                for row in 0..row_size {
                    z[row] = q[ii];
                    ii += iright;
                }

                let zp = z.dot(&m_list[h]);

                for col in 0..row_size {
                    q[ij] = zp[col];
                    ij += iright;
                }
            });
            base_i += row_size * iright;
            base_j += row_size * iright;
        }
        if h + 1 != num_matrices {
            ileft *= row_size;
            iright /= row_size;
        }
    }
    q
}

/// Converts a gate to its matrix representation for a certain circuit structure.
///
/// # Returns
/// * An arrayfire array of complex numbers
///
/// # Arguments
/// * `num_qubits` the size of the quantum circuit
/// * `gate` the gate that needs to be converted
///
pub fn convert_gate_to_operation(
    num_qubits: usize,
    gate_: &Arc<Mutex<dyn gates::Gate>>,
) -> Array2<Complex<f64>> {
    let gate = gate_.lock().unwrap();
    let active_matrix = gate.get_matrix();

    // One qubit gates
    if gate.get_name() == "Identity"
        || gate.get_name() == "Hadamard"
        || gate.get_name() == "RotationX"
        || gate.get_name() == "RotationY"
        || gate.get_name() == "RotationZ"
        || gate.get_name() == "PauliX"
        || gate.get_name() == "PauliY"
        || gate.get_name() == "PauliZ"
    {
        // get active qubit
        let active_qubit = gate.get_target_qubit();

        if num_qubits == 1 {
            return active_matrix.clone();
        }

        let matrix_list = get_list_of_matrices(num_qubits, active_qubit, &active_matrix);

        let mut matrix = matrix_list[0].clone();

        for index in 1..matrix_list.len() {
            matrix = utils::kron(&matrix, &matrix_list[index]);
        }

        return matrix;

    // CNOT
    } else if gate.get_name() == "CNOT" {
        let control_qubit = gate.get_control_qubit();
        let target_qubit = gate.get_target_qubit();

        let (active_matrix_list, inactive_matrix_list) =
            get_cnot_list_of_matrices(num_qubits, control_qubit.unwrap(), target_qubit);

        let mut inactive_matrix = inactive_matrix_list[0].clone();
        for index in 1..inactive_matrix_list.len() {
            inactive_matrix = utils::kron(&inactive_matrix, &inactive_matrix_list[index]);
        }

        // active matrix
        let mut active_matrix = active_matrix_list[0].clone();
        for index in 1..active_matrix_list.len() {
            active_matrix = utils::kron(&active_matrix, &active_matrix_list[index]);
        }

        return inactive_matrix + active_matrix;
    } else {
        panic!("Unsupported gate");
    }
}

/// Converts a gate to its matrix representation for a certain circuit structure.
///
/// # Returns
/// * An arrayfire array of complex numbers
///
/// # Arguments
/// * `num_qubits` the size of the quantum circuit
/// * `gate` the gate that needs to be converted
///
pub fn get_list_of_matrices(
    num_qubits: usize,
    active_qubit: usize,
    active_matrix: &Array2<Complex<f64>>,
) -> Vec<Array2<Complex<f64>>> {
    // build identity matrix
    let _id_gate = gates::Identity::new();
    let _id_matrix = _id_gate.get_matrix();

    let mut matrix_list = Vec::<Array2<Complex<f64>>>::new();

    // fill the matrix list in the right order
    for qubit in (0..num_qubits).rev() {
        if qubit == active_qubit {
            matrix_list.push(active_matrix.to_owned());
        } else {
            matrix_list.push(_id_matrix.to_owned());
        }
    }
    matrix_list
}
pub fn get_cnot_list_of_matrices(
    num_qubits: usize,
    control_qubit: usize,
    target_qubit: usize,
) -> (Vec<Array2<Complex<f64>>>, Vec<Array2<Complex<f64>>>) {
    let id_gate = gates::Identity::new();
    let id_matrix = id_gate.get_matrix();

    // Build projector 0
    let mut inactive_projector = Array2::zeros((2, 2));
    inactive_projector[[0, 0]] = Complex::<f64> { re: 1., im: 0. };

    // Build projector 1
    let mut active_projector = Array2::zeros((2, 2));
    active_projector[[1, 1]] = Complex::<f64> { re: 1., im: 0. };

    // sigma_x
    let x = gates::PauliX::new(0);
    let sigma_x = x.get_matrix();

    let mut inactive_matrix_list = Vec::<Array2<Complex<f64>>>::new();
    let mut active_matrix_list = Vec::<Array2<Complex<f64>>>::new();

    // fill the matrix list in the right order
    for qubit in (0..num_qubits).rev() {
        if qubit == control_qubit {
            inactive_matrix_list.push(inactive_projector.to_owned());
            active_matrix_list.push(active_projector.to_owned());
        } else if qubit == target_qubit {
            inactive_matrix_list.push(id_matrix.to_owned());
            active_matrix_list.push(sigma_x.to_owned());
        } else {
            inactive_matrix_list.push(id_matrix.to_owned());
            active_matrix_list.push(id_matrix.to_owned());
        }
    }
    (active_matrix_list, inactive_matrix_list)
}

/// Samples from a discrete probability distribution from the cumulative
///
/// # Attributes
/// * `cumulative_distribution` cumulative distribution from which to sample
///
/// # Examples
/// ```
/// let cumulative_distribution = vec![0., 0.3, 1.0];
///
/// circuit.sample_from_discrete_distribution(cumulative_distribution);
/// ```
pub fn sample_from_discrete_cumulative(cumulative_distribution: Vec<f64>) -> Option<usize> {
    let mut rng = thread_rng();
    let xsi: f64 = rng.gen::<f64>() * cumulative_distribution.last().unwrap();

    for index in 0..cumulative_distribution.len() {
        if xsi <= cumulative_distribution[index] {
            return Some(index - 1);
        }
    }
    None
}
pub fn sample_from_discrete_distribution(probabilities: &Vec<f64>) -> Option<usize> {
    let mut cumulative_distribution = vec![0.];

    for probability in probabilities {
        cumulative_distribution.push(cumulative_distribution.last().unwrap() + probability);
    }

    sample_from_discrete_cumulative(cumulative_distribution)
}
