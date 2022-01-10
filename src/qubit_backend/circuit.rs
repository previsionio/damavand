//! # circuit
//!
//! `circuit` is a collection of utilities to create quantum circuits.
//!
extern crate libc;

use crate::qubit_backend::gates;
use crate::utils;

#[cfg(feature = "profiling")]
use crate::profiler::Profiler;
#[cfg(feature = "profiling")]
use crate::profiler::OperationType;

use mpi::topology::SystemCommunicator;
use mpi::traits::*;
use ndarray::prelude::*;
use num::complex::Complex;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use sysinfo::{System, SystemExt};


#[cfg(feature = "profiling")]
use std::collections::HashMap;

#[cfg(feature = "gpu")]
#[link(name = "damavand-gpu", kind = "static")]
extern "C" {
    fn get_number_of_available_gpus() -> libc::c_int;
    fn print_timers() -> libc::c_void;

    fn get_memory_for_gpu(gpu_id: libc::c_int) -> libc::c_double;

    fn init_quantum_state(
        num_amplitudes_per_gpu: libc::c_int,
        num_gpus_per_node_required: libc::c_int,
        is_first_node: libc::c_int,
    ) -> libc::c_void;

    fn measure_on_gpu(
        num_amplitudes_per_gpu: libc::c_int,
        probabilities: *mut libc::c_double,
    ) -> *mut libc::c_double;

    fn retrieve_amplitudes_on_host(
        num_amlpitudes_per_gpu: libc::c_int,
        local_amplitudes_real: *mut libc::c_double,
        local_amplitudes_imaginary: *mut libc::c_double,
    ) -> libc::c_void;
}
///
/// struct Circuit
///
/// # Attributes
/// * `num_qubits`: number of qubits on which the circuit will be run
/// * `operations`: the list arrays that represent the quantum operations
/// * `optimizer`: the optimizer to be used to optimize the circuit
///
#[pyclass]
pub struct Circuit {
    pub num_qubits: usize,
    pub operations: Vec<Array2<Complex<f64>>>,
    pub gates: Vec<Arc<Mutex<dyn gates::Gate>>>,
    pub apply_method: utils::ApplyMethod,
    pub local_amplitudes: Array1<Complex<f64>>,
    pub partner_amplitudes: Array1<Complex<f64>>,
    pub num_nodes: usize,
    pub num_amplitudes_per_node: usize,
    pub num_gpus_per_node: usize,
    pub num_amplitudes_per_gpu: usize,
    pub observables: Vec<usize>,

    #[cfg(feature = "profiling")]
    pub profilers: HashMap<OperationType, Profiler>,
}

#[pymethods]
impl Circuit {
    ///
    /// Returns a `Circuit`
    ///
    /// # Arguments
    /// * `num_qubits` number of qubits on which the circuit will be run
    ///
    /// # Examples
    /// ```
    /// use damavand::qubit_backend::circuit::Circuit;
    ///
    /// let _num_qubits = 5;
    /// let circuit = Circuit::new(_num_qubits, Some("brute_force".to_string()));
    /// ```
    #[new]
    pub fn new(num_qubits: usize, apply_method: Option<String>) -> Self {
        let mut apply = utils::ApplyMethod::Multithreading;

        if apply_method.is_some() {
            let input_apply_method = apply_method.unwrap();
            if input_apply_method.eq("brute_force") {
                apply = utils::ApplyMethod::BruteForce;
            } else if input_apply_method.eq("shuffle") {
                apply = utils::ApplyMethod::Shuffle;
            } else if input_apply_method.eq("multithreading") {
                apply = utils::ApplyMethod::Multithreading;
            } else if input_apply_method.eq("gpu") {
                apply = utils::ApplyMethod::GPU;
            } else if input_apply_method.eq("distributed_cpu") {
                apply = utils::ApplyMethod::DistributedCPU;
            } else if input_apply_method.eq("distributed_gpu") {
                apply = utils::ApplyMethod::DistributedGPU;
            } else {
                panic!("Apply method not recognized: {}", input_apply_method);
            }
        }
        let sys = System::new_all();

        let num_nodes = SystemCommunicator::world().size() as usize;
        let safety_size = 1000000_usize;
        let node_size = sys.total_memory() as usize - sys.used_memory() as usize - safety_size;
        let max_num_amplitudes_per_node = node_size * 1000 / (2 * 8) / 2;
        let max_num_qubits_per_node = (max_num_amplitudes_per_node as f64).log(2.0) as usize;

        if num_qubits >= max_num_qubits_per_node {
            panic!("System requires more memory to simulate {} qubits. Currently limited to {} qubits.",
                num_qubits,
                max_num_qubits_per_node * num_nodes,
                )
        }

        #[cfg(feature = "gpu")]
        let num_gpus_available_per_node = unsafe { get_number_of_available_gpus() };
        #[cfg(not(feature = "gpu"))]
        let num_gpus_available_per_node = 0;

        let num_amplitudes = 1_usize << num_qubits;
        let num_amplitudes_per_node = num_amplitudes / num_nodes;

        #[allow(unused_variables, unused_mut)]
        let mut gpu_memory_per_node = 0.;

        #[cfg(feature = "gpu")]
        if [utils::ApplyMethod::GPU, utils::ApplyMethod::DistributedGPU].contains(&apply) {
            if num_gpus_available_per_node == 0 {
                panic!("Could not find any GPU.");
            }
            for gpu_id in 0..num_gpus_available_per_node {
                let memory_per_gpu = unsafe { get_memory_for_gpu(gpu_id) };
                gpu_memory_per_node += memory_per_gpu;
                let num_amplitudes_per_node = gpu_memory_per_node * 1000000. / (2. * 8.) / 2.;
                let num_qubits_per_node = (num_amplitudes_per_node as f64).log(2.0) as usize;

                if num_qubits <= num_qubits_per_node {
                    if gpu_id < num_gpus_available_per_node - 1 {
                        println!("You have allocated more GPUs than necessary for this computation: allocated {}, necessary {}", num_gpus_available_per_node, gpu_id + 1);
                    }
                    break;
                }
            }
        }

        #[allow(unused_mut)]
        let mut num_amplitudes_per_gpu = 0_usize;
        #[allow(unused_assignments)]
        let mut local_amplitudes: Array1<Complex<f64>> = Array::zeros(1);
        #[allow(unused_assignments)]
        let mut partner_amplitudes: Array1<Complex<f64>> = Array::zeros(1);

        match apply {
            utils::ApplyMethod::BruteForce
            | utils::ApplyMethod::Shuffle
            | utils::ApplyMethod::Multithreading => {
                local_amplitudes = Array::zeros((1 << num_qubits) / num_nodes);
                if SystemCommunicator::world().rank() == 0 {
                    local_amplitudes[0] = Complex::<f64> { re: 1., im: 0. };
                }
            }
            utils::ApplyMethod::DistributedCPU => {
                local_amplitudes = Array::zeros((1 << num_qubits) / num_nodes);
                partner_amplitudes = Array::zeros((1 << num_qubits) / num_nodes);

                if SystemCommunicator::world().rank() == 0 {
                    local_amplitudes[0] = Complex::<f64> { re: 1., im: 0. };
                }
            }
            #[cfg(feature = "gpu")]
            utils::ApplyMethod::GPU => {
                num_amplitudes_per_gpu =
                    num_amplitudes_per_node / num_gpus_available_per_node as usize;
                unsafe {
                    init_quantum_state(
                        num_amplitudes_per_gpu as i32,
                        num_gpus_available_per_node,
                        1_i32,
                    )
                };
            }
            #[cfg(feature = "gpu")]
            utils::ApplyMethod::DistributedGPU => {
                num_amplitudes_per_gpu =
                    num_amplitudes_per_node / num_gpus_available_per_node as usize;

                // in the case of mulple nodes, we need to allocate local and partner amplitudes on the
                // host in order to transfer data between nodes and then push them back one each GPU.
                if num_nodes > 1 {
                    local_amplitudes = Array::zeros(num_amplitudes_per_node);
                    partner_amplitudes = Array::zeros(num_amplitudes_per_node);
                }
                if SystemCommunicator::world().rank() == 0 {
                    unsafe {
                        init_quantum_state(
                            num_amplitudes_per_gpu as i32,
                            num_gpus_available_per_node,
                            1_i32,
                        )
                    };
                } else {
                    unsafe {
                        init_quantum_state(
                            num_amplitudes_per_gpu as i32,
                            num_gpus_available_per_node,
                            0_i32,
                        )
                    };
                }
            },
            #[allow(unreachable_patterns)]
            _ => {}
        };

        #[cfg(feature = "profiling")]
        let mut profilers = HashMap::new();
        #[cfg(feature = "profiling")]
        profilers.insert(OperationType::Forward, Profiler::new());
        #[cfg(feature = "profiling")]
        profilers.insert(OperationType::InterNodeCommunication, Profiler::new());
        #[cfg(feature = "profiling")]
        profilers.insert(OperationType::InterGPUCommunication, Profiler::new());
        #[cfg(feature = "profiling")]
        profilers.insert(OperationType::Sampling, Profiler::new());


        Circuit {
            num_qubits: num_qubits as usize,
            operations: Vec::<Array2<Complex<f64>>>::new(),
            gates: Vec::<Arc<Mutex<dyn gates::Gate>>>::new(),
            apply_method: apply,
            local_amplitudes: local_amplitudes,
            partner_amplitudes: partner_amplitudes,
            num_nodes: num_nodes,
            num_amplitudes_per_node: num_amplitudes_per_node,
            num_gpus_per_node: num_gpus_available_per_node as usize,
            num_amplitudes_per_gpu: num_amplitudes_per_gpu,
            observables: vec![],

            #[cfg(feature = "profiling")]
            profilers: profilers,

        }
    }

    /// Resets the quantum state to zero state
    pub fn reset(&mut self) {
        self.local_amplitudes =
            Array1::<Complex<f64>>::zeros((1 << self.num_qubits) / self.num_nodes);
        if SystemCommunicator::world().rank() == 0 {
            self.local_amplitudes[0] = Complex::<f64> { re: 1., im: 0. };
        }
        self.partner_amplitudes =
            Array1::<Complex<f64>>::zeros((1 << self.num_qubits) / self.num_nodes);

        #[cfg(feature = "gpu")]
        if self.apply_method == utils::ApplyMethod::GPU {
            unsafe {
                init_quantum_state(
                    self.num_amplitudes_per_gpu as i32,
                    self.num_gpus_per_node as i32,
                    1_i32,
                )
            };
        }

        #[cfg(feature = "gpu")]
        if self.apply_method == utils::ApplyMethod::DistributedGPU {
            if SystemCommunicator::world().rank() == 0 {
                unsafe {
                    init_quantum_state(
                        self.num_amplitudes_per_gpu as i32,
                        self.num_gpus_per_node as i32,
                        1_i32,
                    )
                };
            } else {
                unsafe {
                    init_quantum_state(
                        self.num_amplitudes_per_gpu as i32,
                        self.num_gpus_per_node as i32,
                        0_i32,
                    )
                };
            }
        }
        self.gates = vec![];
    }

    pub fn set_parameters(&mut self, parameters: Vec<f64>) {

        let mut parametrized_gates: Vec<usize> = vec![];
        for gate_index in 0..self.gates.len() {
            let gate = self.gates[gate_index].lock().unwrap();
            if gate.get_parameter().is_some(){
                parametrized_gates.push(gate_index);
            }
        }

        for (gate_index, parameter) in parametrized_gates.into_iter().zip(&parameters) {
            let mut gate = self.gates[gate_index].lock().unwrap();
            gate.set_parameter(*parameter);
        }
    }

    /// Forward method: implements a forward pass of a quantum state through a quantum circuit
    /// until measurment.
    ///
    /// # Examples
    /// ```
    /// use damavand::qubit_backend::circuit::Circuit;
    ///
    /// // Create a circuit with 5 qubits
    /// let num_qubits = 1;
    /// let mut circuit = Circuit::new(num_qubits, Some("brute_force".to_string()));
    ///
    /// // Add a Hadamard gate on the first qubit
    /// circuit.add_hadamard_gate(0);
    ///
    /// // Propagate the state through the circuit
    /// circuit.forward();
    /// ```
    pub fn forward(&mut self) {

        #[cfg(feature = "profiling")]
        self.start_profiling_forward();

        for gate_index in 0..self.gates.len() {
            if self.observables.contains(&gate_index) {
                continue;
            }
            match self.apply_method {
                utils::ApplyMethod::BruteForce => self.apply_brute_force(gate_index),
                utils::ApplyMethod::Shuffle => self.apply_shuffle(gate_index),
                utils::ApplyMethod::Multithreading => {
                    self.apply_multithreading_local(gate_index);
                }
                utils::ApplyMethod::GPU => {
                    #[cfg(feature = "gpu")]
                    self.apply_gpu_local(gate_index);
                }
                utils::ApplyMethod::DistributedCPU => {
                    self.apply_distributed_cpu(gate_index);
                }
                utils::ApplyMethod::DistributedGPU => {
                    #[cfg(feature = "gpu")]
                    self.apply_distributed_gpu(gate_index);
                }
            }
        }

        #[cfg(feature = "profiling")]
        self.stop_profiling_forward();

        #[cfg(feature = "gpu")]
        self.retrieve_amplitudes_on_host();
    }

    /// Retrieves amplitudes on host when the computation was perfromed on GPUs.
    /// until measurment.
    #[cfg(feature = "gpu")]
    pub fn retrieve_amplitudes_on_host(&mut self) {
        match self.apply_method {
            utils::ApplyMethod::GPU | utils::ApplyMethod::DistributedGPU => {
                let mut local_real = vec![0.; self.num_amplitudes_per_node];
                let mut local_imag = vec![0.; self.num_amplitudes_per_node];

                unsafe {
                    retrieve_amplitudes_on_host(
                        self.num_amplitudes_per_gpu as i32,
                        local_real.as_mut_ptr(),
                        local_imag.as_mut_ptr(),
                    );
                }

                self.local_amplitudes = Array::zeros(self.num_amplitudes_per_node);

                for i in 0..self.num_amplitudes_per_node {
                    self.local_amplitudes[i] = Complex::<f64> {
                        re: local_real[i],
                        im: local_imag[i],
                    };
                }
            }
            _ => {}
        }
    }

    /// Sample the circuit.
    ///
    /// # Returns
    /// `Array<f64>`: an array of probabilities associated to each quantum amplitude
    ///
    /// # Examples
    /// ```
    /// use damavand::qubit_backend::circuit::Circuit;
    ///
    /// // Create a circuit with 5 qubits.
    /// let num_qubits = 5;
    /// let mut circuit = Circuit::new(num_qubits, Some("brute_force".to_string()));
    /// let quantum_state = State::new(num_qubits);
    ///
    /// // Add a Hadamard gate on the first qubit
    /// circuit.add_hadamard_gate(0);
    ///
    /// // Propagate the state through the circuit
    /// circuit.forward();
    ///
    /// // Measures last updates quantum state.
    /// circuit.measure();
    ///
    /// // sample the circuit num_samples times
    /// let num_samples = 1024
    /// circuit.sample(Some(num_sampes));
    /// ```
    pub fn sample(&mut self, num_samples: Option<usize>) -> Vec<usize> {

        #[cfg(feature = "profiling")]
        self.start_profiling_sampling();

        let num_samples = if num_samples.is_some() {
            num_samples.unwrap()
        } else {
            1000
        };
        let node_probabilities = self.measure();

        let samples = if self.apply_method == utils::ApplyMethod::DistributedCPU
            || self.apply_method == utils::ApplyMethod::DistributedGPU
        {
            self.sample_distributed(num_samples, node_probabilities)
        } else {
            self.sample_local(num_samples, node_probabilities)
        };

        #[cfg(feature = "profiling")]
        self.stop_profiling_sampling();

        samples
    }

    /// Sample the circuit when there is only one node.
    ///
    /// # Arguments:
    /// `num_samples`: number of samples to be drawn
    /// `node_probabilities`: probability distribution from which to draw the samples
    ///
    /// # Returns
    /// `Array<f64>`: an array of probabilities associated to each quantum amplitude
    pub fn sample_local(
        &mut self,
        num_samples: usize,
        node_probabilities: Vec<f64>,
    ) -> Vec<usize> {
        let mut samples = vec![];

        for _ in 0..num_samples {
            let sample_index = utils::sample_from_discrete_distribution(&node_probabilities);

            if sample_index.is_some() {
                samples.push(sample_index.unwrap());
            } else {
                panic!("Could not sample from output distribution.");
            }
        }
        samples
    }

    /// Extract the expectation value of observables from samples
    ///
    /// # Arguments
    /// `samples`: samples drawn from calculation
    ///
    /// # Returns
    /// `Vec<Vec<f64>>`: a vector containing the expectation values of the observables
    pub fn extract_expectation_values(&self, samples: Vec<usize>) -> Vec<Vec<f64>> {
        let mut expectation_values = vec![];
        for sample in &samples {
            let mut observable_expectation_values = vec![];
            for observable_index in self.observables.clone() {
                let readout_qubit = self.gates[observable_index]
                        .lock()
                        .unwrap()
                        .get_target_qubit();
                let bit_value = (sample >> readout_qubit) & 1;
                if bit_value > 0 {
                    observable_expectation_values.push(-1.);
                } else {
                    observable_expectation_values.push(1.);
                }
            }
            expectation_values.push(observable_expectation_values);
        }
        expectation_values
    }

    /// Compute the gradient of one parameter gate by applying the parameter shift rule
    ///
    /// # Attributes
    /// * `gate_index` index of the gate that needs to be applied.
    ///
    /// # Examples
    /// ```
    /// use damavand::qubit_backend::circuit::Circuit;
    ///
    /// let num_qubits = 3;
    /// let circuit = Circuit::new(num_qubits);
    ///
    /// circuit.backward(0);
    /// ```
    // pub fn backward(&mut self, gate_index: usize) {
    //     for gate_ in &mut self.gates {
    //         let gate = gate_.lock().unwrap();
    //         if !gate.get_parameter().is_some() {
    //             continue;
    //         }
    //         gate.set_parameter(gate.get_parameter().unwrap() + std::f64::consts::PI / 2.);
    //         let samples = self.sample();
    //         let upper_gradient = samples.iter().sum::<f64>() / samples.len();

    //         gate.set_parameter(gate.get_parameter().unwrap() - std::f64::consts::PI);
    //         let samples = self.sample();
    //         let lower_gradient = samples.iter().sum::<f64>() / samples.len();

    //         let gradient = 0.5 * (upper_gradient - lower_gradient);
    //         drop(gate);
    //     }
    // }

    /// Measures a quantum state. Usually run at the end of the forward pass so as to collapse the
    /// output of a quantum circuit.
    ///
    /// # Examples
    /// ```
    /// use damavand::qubit_backend::circuit::Circuit;
    ///
    /// // Create a circuit with 5 qubits.
    /// let num_qubits = 5;
    /// let mut circuit = Circuit::new(num_qubits, Some("brute_force".to_string()));
    ///
    /// circuit.add_hadamard_gate(0);
    /// circuit.forward();
    ///
    /// // Measures last updates quantum state.
    /// circuit.measure();
    /// ```
    pub fn measure(&self) -> Vec<f64> {
        let mut probabilities = vec![0_f64; self.num_amplitudes_per_node];

        if self.apply_method == utils::ApplyMethod::GPU
            || self.apply_method == utils::ApplyMethod::DistributedGPU
        {
            #[cfg(feature = "gpu")]
            unsafe {
                measure_on_gpu(
                    self.num_amplitudes_per_gpu as i32,
                    probabilities.as_mut_ptr(),
                );
            }
        } else {
            for i in 0..self.num_amplitudes_per_node {
                probabilities[i] = self.local_amplitudes[i].norm_sqr();
            }
        }
        #[cfg(feature = "gpu")]
        #[cfg(feature = "profiling")]
        unsafe {
            print_timers();
        }
        probabilities
    }

    /// Retrieve a vector with the real part of the amplitudes of the state.
    /// Usually run at the end of a run.
    ///
    /// # Returns:
    /// `Vec<f64>` a vector containint the real parts of the amplitudes.
    pub fn get_real_part_state(&self) -> Vec<f64> {
        self.local_amplitudes.iter().map(|a| a.re).collect()
    }

    pub fn get_imaginary_part_state(&self) -> Vec<f64> {
        self.local_amplitudes.iter().map(|a| a.im).collect()
    }

    /// Python wrapper to add a Hadamard gate
    ///
    /// # Arguments
    /// * `active_qubit` the qubit on which the Hadamard gate needs to be applied
    pub fn add_hadamard_gate(&mut self, active_qubit: usize) {
        self.gates
            .push(Arc::new(Mutex::new(gates::Hadamard::new(active_qubit))));
    }

    /// Python wrapper to add a RotationX gate
    ///
    /// # Arguments
    /// * `active_qubit` the qubit on which the RotationX gate needs to be applied
    pub fn add_rotation_x_gate(&mut self, active_qubit: usize, theta: f64) {
        self.gates.push(Arc::new(Mutex::new(gates::RotationX::new(
            active_qubit,
            theta,
        ))));
    }

    /// Python wrapper to add a RotationY gate
    ///
    /// # Arguments
    /// * `active_qubit` the qubit on which the RotationY gate needs to be applied
    pub fn add_rotation_y_gate(&mut self, active_qubit: usize, theta: f64) {
        self.gates.push(Arc::new(Mutex::new(gates::RotationY::new(
            active_qubit,
            theta,
        ))));
    }

    /// Python wrapper to add a RotationZ gate
    ///
    /// # Arguments
    /// * `active_qubit` the qubit on which the RotationZ gate needs to be applied
    pub fn add_rotation_z_gate(&mut self, active_qubit: usize, theta: f64) {
        self.gates.push(Arc::new(Mutex::new(gates::RotationZ::new(
            active_qubit,
            theta,
        ))));
    }
    /// Python wrapper to add a PauliX gate
    ///
    /// # Arguments
    /// * `active_qubit` the qubit on which the PauliX gate needs to be applied
    pub fn add_pauli_x_gate(&mut self, active_qubit: usize, is_observable: bool) {
        self.gates
            .push(Arc::new(Mutex::new(gates::PauliX::new(active_qubit))));
        if is_observable {
            self.observables.push(self.gates.len() - 1);
        }
    }

    /// Python wrapper to add a PauliY gate
    ///
    /// # Arguments
    /// * `active_qubit` the qubit on which the RotationY gate needs to be applied
    pub fn add_pauli_y_gate(&mut self, active_qubit: usize, is_observable: bool) {
        self.gates
            .push(Arc::new(Mutex::new(gates::PauliY::new(active_qubit))));
        if is_observable {
            self.observables.push(self.gates.len() - 1);
        }
    }

    /// Python wrapper to add a PauliZ gate
    ///
    /// # Arguments
    /// * `active_qubit` the qubit on which the PauliZ gate needs to be applied
    pub fn add_pauli_z_gate(&mut self, active_qubit: usize, is_observable: bool) {
        self.gates
            .push(Arc::new(Mutex::new(gates::PauliZ::new(active_qubit))));
        if is_observable {
            self.observables.push(self.gates.len() - 1);
        }
    }

    /// Python wrapper to add a CNOT gate
    ///
    /// # Arguments
    /// * `control_qubit` control qubit of the CNOT gate
    /// * `target_qubit` target qubit of the CNOT gate
    pub fn add_cnot_gate(&mut self, control_qubit: usize, target_qubit: usize) {
        self.gates.push(Arc::new(Mutex::new(gates::CNOT::new(
            control_qubit,
            target_qubit,
        ))));
    }

    /// Prints operations' matrices
    #[allow(unused)]
    pub fn print_operations(&self) {
        for operation in &self.operations {
            println!("{}", operation);
        }
    }
    #[cfg(feature = "profiling")]
    pub fn print_profiling_results(&self) {

        println!("profiling results");
        println!("profiling Forward: iterations {} elapsed time {}",
                 self.profilers.get(&OperationType::Forward).unwrap().iterations,
                 self.get_mean_elapsed_forward());

        println!("profiling InterNodeCommunications: iterations {} elapsed time {}",
                 self.profilers.get(&OperationType::InterNodeCommunication).unwrap().iterations,
                 self.get_mean_elapsed_inter_node_communications());

        println!("profiling InterGPUCommunications: iterations {} elapsed time {}",
                 self.profilers.get(&OperationType::InterGPUCommunication).unwrap().iterations,
                 self.get_mean_elapsed_inter_gpu_communications());

        println!("profiling Sampling: iterations {} elapsed time {}",
                 self.profilers.get(&OperationType::Sampling).unwrap().iterations,
                 self.get_mean_elapsed_sampling());
    }

    #[cfg(feature = "profiling")]
    pub fn get_profiling_results_forward(&self) -> (usize, f64) {
        (
            self.profilers.get(&OperationType::Forward).unwrap().iterations,
            self.get_mean_elapsed_forward()
        )
    }

    #[cfg(feature = "profiling")]
    pub fn get_profiling_results_inter_node_communications(&self) -> (usize, f64) {
        (
            self.profilers.get(&OperationType::InterNodeCommunication).unwrap().iterations,
            self.get_mean_elapsed_inter_node_communications()
        )
    }

    #[cfg(feature = "profiling")]
    pub fn get_profiling_results_inter_gpu_communications(&self) -> (usize, f64) {
        (
            self.profilers.get(&OperationType::InterGPUCommunication).unwrap().iterations,
            self.get_mean_elapsed_inter_gpu_communications()
        )
    }

    #[cfg(feature = "profiling")]
    pub fn get_profiling_results_sampling(&self) -> (usize, f64) {
        (
            self.profilers.get(&OperationType::Sampling).unwrap().iterations,
            self.get_mean_elapsed_sampling()
        )
    }

    pub fn get_fidelity_between_two_states_with_parameters(
        &mut self,
        parameters_1: Vec<f64>,
        parameters_2: Vec<f64>
    ) -> f64 {
        self.reset();
        self.set_parameters(parameters_1);
        self.forward();
        let state_1 = self.local_amplitudes.clone();

        self.reset();
        self.set_parameters(parameters_2);
        self.forward();
        let state_2 = self.local_amplitudes.clone();

        self.get_fidelity(state_1, state_2)
    }
}
impl Circuit {
    /// Computes the partner rank of a given amplitude rank.
    ///
    /// # Arguments
    /// `current_node_rank`: the rank of the amplitude for which we want the partner.
    /// `num_amplitudes_per_node`: number of amplitudes per node.
    /// `amplitude_gap`: the amplitude gab between the two amplitudes
    ///
    /// # Returns:
    /// `usize` the partner rank
    pub fn compute_partner_rank(
        current_node_rank: usize,
        num_amplitudes_per_node: usize,
        amplitude_gap: usize,
    ) -> usize {
        let num_amplitudes_before_node = current_node_rank * num_amplitudes_per_node;

        if num_amplitudes_before_node % (2 * amplitude_gap) < amplitude_gap {
            // node is linked with a node of higher rank
            current_node_rank + amplitude_gap / num_amplitudes_per_node
        } else {
            // node is linked with a node of lower rank
            current_node_rank - amplitude_gap / num_amplitudes_per_node
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_brute_force() {
        let _num_qubits = 2;
        let mut circuit = Circuit::new(_num_qubits, Some("brute_force".to_string()));
        circuit.add_hadamard_gate(0);
        circuit.add_hadamard_gate(1);
        circuit.add_cnot_gate(0, 1);
        circuit.forward();
        // println!("{}", circuit.local_amplitudes);
        let expected_local_amplitudes = array![
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            }
        ];
        for (item, expected) in circuit
            .local_amplitudes
            .iter()
            .zip(&expected_local_amplitudes)
        {
            let diff = item - expected;
            assert!(diff.norm() < 0.01);
        }
    }

    #[test]
    fn test_apply_shuffle() {
        let _num_qubits = 2;
        let mut circuit = Circuit::new(_num_qubits, Some("shuffle".to_string()));
        circuit.add_hadamard_gate(0);
        circuit.add_hadamard_gate(1);
        circuit.add_cnot_gate(0, 1);
        circuit.forward();
        // println!("{}", circuit.local_amplitudes);
        let expected_local_amplitudes = array![
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            }
        ];
        for (item, expected) in circuit
            .local_amplitudes
            .iter()
            .zip(&expected_local_amplitudes)
        {
            let diff = item - expected;
            assert!(diff.norm() < 0.01);
        }
    }
    #[test]
    fn test_apply_element_wise() {
        let _num_qubits = 2;
        let mut circuit = Circuit::new(_num_qubits, Some("element_wise".to_string()));
        circuit.add_hadamard_gate(0);
        circuit.add_hadamard_gate(1);
        circuit.add_cnot_gate(0, 1);
        circuit.forward();
        // println!("{}", circuit.local_amplitudes);
        let expected_local_amplitudes = array![
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            }
        ];
        // for (item, expected) in circuit.local_amplitudes.iter().zip(&expected_local_amplitudes) {
        //     let diff = item - expected;
        //     assert!(diff.norm() < 0.01);
        // }
    }

    #[test]
    fn test_apply_gpu() {
        unsafe {
            println!("Number of GPUs found: {}", get_number_of_available_gpus());
        }
        let _num_qubits = 2;
        let mut circuit = Circuit::new(_num_qubits, Some("gpu".to_string()));
        circuit.add_hadamard_gate(0);
        circuit.add_hadamard_gate(1);
        circuit.add_cnot_gate(0, 1);
        circuit.forward();
        // println!("{}", circuit.local_amplitudes);
        let expected_local_amplitudes = array![
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            }
        ];
        // for (item, expected) in circuit.local_amplitudes.iter().zip(&expected_local_amplitudes) {
        //     let diff = item - expected;
        //     assert!(diff.norm() < 0.01);
        // }
    }
    #[test]
    fn test_apply_distributed_cpu() {
        let _num_qubits = 2;
        let mut circuit = Circuit::new(_num_qubits, Some("distributed_cpu".to_string()));
        // circuit.add_hadamard_gate(0);
        circuit.add_hadamard_gate(1);
        // circuit.add_cnot_gate(0, 1);
        circuit.forward();
        // println!("{}", circuit.local_amplitudes);
        let expected_local_amplitudes = array![
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            },
            Complex::<f64> {
                re: 1. / 2.,
                im: 0.
            }
        ];
        // for (item, expected) in circuit.local_amplitudes.iter().zip(&expected_local_amplitudes) {
        //     let diff = item - expected;
        //     assert!(diff.norm() < 0.01);
        // }
    }
}
