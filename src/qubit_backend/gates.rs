//! # gate
//!
//! `model` is a collection of utilities to create quantum gates.
//!
use ndarray::prelude::*;
use num::complex::Complex;
use pyo3::prelude::*;

pub trait Observable {
    fn get_eigenvalues(&self) -> Vec<f64>;
}

pub trait Gate: Send {
    fn print(&self);
    fn get_name(&self) -> &String;
    fn get_matrix(&self) -> Array2<Complex<f64>>;
    fn get_control_qubit(&self) -> Option<usize>;
    fn get_target_qubit(&self) -> usize;
    fn get_parameter(&self) -> Option<f64>;
    fn set_parameter(&mut self, parameter: f64);
}

/// struct Identity
///
/// # Attributes
/// * `name`: name of the gate
/// * `gate_matrix`: ArrayFire array representing the basic gate_matrix.
///
#[pyclass]
pub struct Identity {
    name: String,
    pub gate_matrix: Array2<Complex<f64>>,
}

#[pymethods]
impl Identity {
    /// Returns the Identity gate
    #[new]
    pub fn new() -> Identity {
        let gate_matrix = array![
            [
                Complex::<f64> { re: 1., im: 0. },
                Complex::<f64> { re: 0., im: 0. }
            ],
            [
                Complex::<f64> { re: 0., im: 0. },
                Complex::<f64> { re: 1., im: 0. }
            ]
        ];
        Identity {
            name: "Identity".to_string(),
            gate_matrix: gate_matrix,
        }
    }
}
impl Gate for Identity {
    fn print(&self) {
        println!("Identity");
    }
    fn get_name(&self) -> &String {
        &self.name
    }
    fn get_matrix(&self) -> Array2<Complex<f64>> {
        self.gate_matrix.clone()
    }
    fn get_control_qubit(&self) -> Option<usize> {
        None
    }
    fn get_target_qubit(&self) -> usize {
        panic!("get_target_qubit() should not be called for Identity gates.");
    }
    fn get_parameter(&self) -> Option<f64> {
        None
    }
    fn set_parameter(&mut self, _parameter: f64) {
        panic!("Cannot set parameter to Identity gate.")
    }
}

/// struct Hadamard
///
/// # Attributes
/// * `name`: name of the gate
/// * `gate_matrix`: ArrayFire array representing the basic gate_matrix.
/// * `active_qubit`: Qubit id on which the Hadamard gate is applied
///
#[pyclass]
pub struct Hadamard {
    name: String,
    gate_matrix: Array2<Complex<f64>>,
    active_qubit: usize,
}

#[pymethods]
impl Hadamard {
    #[new]
    pub fn new(active_qubit: usize) -> Hadamard {
        let gate_matrix = array![
            [
                Complex::<f64> {
                    re: 1. / (2.0 as f64).sqrt(),
                    im: 0.
                },
                Complex::<f64> {
                    re: 1. / (2.0 as f64).sqrt(),
                    im: 0.
                }
            ],
            [
                Complex::<f64> {
                    re: 1. / (2.0 as f64).sqrt(),
                    im: 0.
                },
                Complex::<f64> {
                    re: -1. / (2.0 as f64).sqrt(),
                    im: 0.
                }
            ]
        ];
        Hadamard {
            name: "Hadamard".to_string(),
            gate_matrix: gate_matrix,
            active_qubit: active_qubit,
        }
    }
}
impl Gate for Hadamard {
    fn print(&self) {
        println!("Hadamard");
    }
    fn get_name(&self) -> &String {
        &self.name
    }
    fn get_matrix(&self) -> Array2<Complex<f64>> {
        self.gate_matrix.clone()
    }
    fn get_control_qubit(&self) -> Option<usize> {
        None
    }
    fn get_target_qubit(&self) -> usize {
        self.active_qubit
    }
    fn get_parameter(&self) -> Option<f64> {
        None
    }
    fn set_parameter(&mut self, _parameter: f64) {
        panic!("Cannot set parameter to Identity gate.")
    }
}

/// struct CNOT
///
/// # Attributes
/// * `name`: name of the gate
/// * `gate_matrix`: ArrayFire array representing the basic gate_matrix.
/// * `control_qubit`: Control qubit of the CNOT gate.
/// * `target_qubit`: Target qubit of the CNOT gate.
///
#[pyclass]
pub struct CNOT {
    name: String,
    gate_matrix: Array2<Complex<f64>>,
    control_qubit: usize,
    target_qubit: usize,
}
#[pymethods]
impl CNOT {
    #[new]
    pub fn new(control_qubit: usize, target_qubit: usize) -> CNOT {
        let mut gate_matrix = Array2::<Complex<f64>>::zeros([2, 2]);
        gate_matrix[[0, 0]] = Complex::<f64> { re: 0., im: 0. };
        gate_matrix[[0, 1]] = Complex::<f64> { re: 1., im: 0. };
        gate_matrix[[1, 0]] = Complex::<f64> { re: 1., im: 0. };
        gate_matrix[[1, 1]] = Complex::<f64> { re: 0., im: 0. };
        CNOT {
            name: "CNOT".to_string(),
            gate_matrix: gate_matrix,
            control_qubit: control_qubit,
            target_qubit: target_qubit,
        }
    }
}
impl Gate for CNOT {
    fn print(&self) {
        println!("CNOT");
    }
    fn get_name(&self) -> &String {
        &self.name
    }
    fn get_matrix(&self) -> Array2<Complex<f64>> {
        self.gate_matrix.clone()
    }
    fn get_control_qubit(&self) -> Option<usize> {
        Some(self.control_qubit)
    }
    fn get_target_qubit(&self) -> usize {
        self.target_qubit
    }
    fn get_parameter(&self) -> Option<f64> {
        None
    }
    fn set_parameter(&mut self, _parameter: f64) {
        panic!("Cannot set parameter to CNOT gate.")
    }
}

/// struct PauliX
///
/// # Attributes
/// * `name`: name of the gate
/// * `gate_matrix`: ArrayFire array representing the basic gate_matrix.
/// * `active_qubit`: Qubit id on which the PauliX gate is applied
///
#[pyclass]
pub struct PauliX {
    name: String,
    gate_matrix: Array2<Complex<f64>>,
    active_qubit: usize,
}
#[pymethods]
impl PauliX {
    #[new]
    pub fn new(active_qubit: usize) -> PauliX {
        let gate_matrix = array![
            [
                Complex::<f64> { re: 0., im: 0. },
                Complex::<f64> { re: 1., im: 0. }
            ],
            [
                Complex::<f64> { re: 1., im: 0. },
                Complex::<f64> { re: 0., im: 0. }
            ]
        ];

        PauliX {
            name: "PauliX".to_string(),
            gate_matrix: gate_matrix,
            active_qubit: active_qubit,
        }
    }
}
impl Gate for PauliX {
    fn print(&self) {
        println!("PauliX");
    }
    fn get_name(&self) -> &String {
        &self.name
    }
    fn get_matrix(&self) -> Array2<Complex<f64>> {
        self.gate_matrix.clone()
    }
    fn get_control_qubit(&self) -> Option<usize> {
        None
    }
    fn get_target_qubit(&self) -> usize {
        self.active_qubit
    }
    fn get_parameter(&self) -> Option<f64> {
        None
    }
    fn set_parameter(&mut self, _parameter: f64) {
        panic!("Cannot set parameter to PauliX gate.")
    }
}

impl Observable for PauliX {
    fn get_eigenvalues(&self) -> Vec<f64> {
        vec![-1., 1.]
    }
}

/// struct PauliY
///
/// # Attributes
/// * `name`: name of the gate
/// * `gate_matrix`: ArrayFire array representing the basic gate_matrix.
/// * `active_qubit`: Qubit id on which the PauliY gate is applied
///
#[pyclass]
pub struct PauliY {
    name: String,
    gate_matrix: Array2<Complex<f64>>,
    active_qubit: usize,
}
#[pymethods]
impl PauliY {
    #[new]
    pub fn new(active_qubit: usize) -> PauliY {
        let gate_matrix = array![
            [
                Complex::<f64> { re: 0., im: 0. },
                Complex::<f64> { re: 0., im: -1. }
            ],
            [
                Complex::<f64> { re: 0., im: 1. },
                Complex::<f64> { re: 0., im: 0. }
            ]
        ];
        PauliY {
            name: "PauliY".to_string(),
            gate_matrix: gate_matrix,
            active_qubit: active_qubit,
        }
    }
}
impl Gate for PauliY {
    fn print(&self) {
        println!("PauliY");
    }
    fn get_name(&self) -> &String {
        &self.name
    }
    fn get_matrix(&self) -> Array2<Complex<f64>> {
        self.gate_matrix.clone()
    }
    fn get_control_qubit(&self) -> Option<usize> {
        None
    }
    fn get_target_qubit(&self) -> usize {
        self.active_qubit
    }
    fn get_parameter(&self) -> Option<f64> {
        None
    }
    fn set_parameter(&mut self, _parameter: f64) {
        panic!("Cannot set parameter to PauliY gate.")
    }
}

impl Observable for PauliY {
    fn get_eigenvalues(&self) -> Vec<f64> {
        vec![-1., 1.]
    }
}

/// struct PauliZ
///
/// # Attributes
/// * `name`: name of the gate
/// * `gate_matrix`: ArrayFire array representing the basic gate_matrix.
/// * `active_qubit`: Qubit id on which the PauliZ gate is applied
///
#[pyclass]
pub struct PauliZ {
    name: String,
    gate_matrix: Array2<Complex<f64>>,
    active_qubit: usize,
}
#[pymethods]
impl PauliZ {
    #[new]
    pub fn new(active_qubit: usize) -> PauliZ {
        let gate_matrix = array![
            [
                Complex::<f64> { re: 1., im: 0. },
                Complex::<f64> { re: 0., im: 0. }
            ],
            [
                Complex::<f64> { re: 0., im: 0. },
                Complex::<f64> { re: -1., im: 0. }
            ]
        ];
        PauliZ {
            name: "PauliZ".to_string(),
            gate_matrix: gate_matrix,
            active_qubit: active_qubit,
        }
    }
}
impl Gate for PauliZ {
    fn print(&self) {
        println!("PauliZ");
    }
    fn get_name(&self) -> &String {
        &self.name
    }
    fn get_matrix(&self) -> Array2<Complex<f64>> {
        self.gate_matrix.clone()
    }
    fn get_control_qubit(&self) -> Option<usize> {
        None
    }
    fn get_target_qubit(&self) -> usize {
        self.active_qubit
    }
    fn get_parameter(&self) -> Option<f64> {
        None
    }
    fn set_parameter(&mut self, _parameter: f64) {
        panic!("Cannot set parameter to PauliZ gate.")
    }
}

impl Observable for PauliZ {
    fn get_eigenvalues(&self) -> Vec<f64> {
        vec![-1., 1.]
    }
}

/// struct RotationX
///
/// # Attributes
/// * `name`: name of the gate
/// * `gate_matrix`: ArrayFire array representing the basic gate_matrix.
/// * `active_qubit`: Qubit id on which the RotationX gate is applied
///
#[pyclass]
pub struct RotationX {
    name: String,
    gate_matrix: Array2<Complex<f64>>,
    active_qubit: usize,
    parameter: f64,
}
#[pymethods]
impl RotationX {
    #[new]
    pub fn new(active_qubit: usize, theta: f64) -> RotationX {
        let gate_matrix = array![
            [
                Complex::<f64> {
                    re: (theta / 2.).cos(),
                    im: 0.
                },
                Complex::<f64> {
                    re: 0.,
                    im: - (theta / 2.).sin()
                }
            ],
            [
                Complex::<f64> {
                    re: 0.,
                    im: - (theta / 2.).sin()
                },
                Complex::<f64> {
                    re: (theta / 2.).cos(),
                    im: 0.
                }
            ]
        ];
        RotationX {
            name: "RotationX".to_string(),
            gate_matrix: gate_matrix,
            active_qubit: active_qubit,
            parameter: theta,
        }
    }
}
impl Gate for RotationX {
    fn print(&self) {
        println!("RotationX");
    }
    fn get_name(&self) -> &String {
        &self.name
    }
    fn get_matrix(&self) -> Array2<Complex<f64>> {
        self.gate_matrix.clone()
    }
    fn get_control_qubit(&self) -> Option<usize> {
        None
    }
    fn get_target_qubit(&self) -> usize {
        self.active_qubit
    }
    fn get_parameter(&self) -> Option<f64> {
        Some(self.parameter)
    }
    fn set_parameter(&mut self, parameter: f64) {
        *self = Self::new(self.active_qubit, parameter);
    }
}

/// struct RotationY
///
/// # Attributes
/// * `name`: name of the gate
/// * `gate_matrix`: ArrayFire array representing the basic gate_matrix.
/// * `active_qubit`: Qubit id on which the RotationY gate is applied
///
#[pyclass]
pub struct RotationY {
    name: String,
    gate_matrix: Array2<Complex<f64>>,
    active_qubit: usize,
    parameter: f64,
}
#[pymethods]
impl RotationY {
    #[new]
    pub fn new(active_qubit: usize, theta: f64) -> RotationY {
        let gate_matrix = array![
            [
                Complex::<f64> {
                    re: (theta / 2.).cos(),
                    im: 0.
                },
                Complex::<f64> {
                    re: -(theta / 2.).sin(),
                    im: 0.
                }
            ],
            [
                Complex::<f64> {
                    re: (theta / 2.).sin(),
                    im: 0.
                },
                Complex::<f64> {
                    re: (theta / 2.).cos(),
                    im: 0.
                }
            ]
        ];
        RotationY {
            name: "RotationY".to_string(),
            gate_matrix: gate_matrix,
            active_qubit: active_qubit,
            parameter: theta,
        }
    }
}
impl Gate for RotationY {
    fn print(&self) {
        println!("RotationY");
    }
    fn get_name(&self) -> &String {
        &self.name
    }
    fn get_matrix(&self) -> Array2<Complex<f64>> {
        self.gate_matrix.clone()
    }
    fn get_control_qubit(&self) -> Option<usize> {
        None
    }
    fn get_target_qubit(&self) -> usize {
        self.active_qubit
    }
    fn get_parameter(&self) -> Option<f64> {
        Some(self.parameter)
    }
    fn set_parameter(&mut self, parameter: f64) {
        *self = Self::new(self.active_qubit, parameter);
    }
}

/// struct RotationZ
///
/// # Attributes
/// * `name`: name of the gate
/// * `gate_matrix`: ArrayFire array representing the basic gate_matrix.
/// * `active_qubit`: Qubit id on which the RotationZ gate is applied
///
#[pyclass]
pub struct RotationZ {
    name: String,
    gate_matrix: Array2<Complex<f64>>,
    active_qubit: usize,
    parameter: f64,
}
#[pymethods]
impl RotationZ {
    #[new]
    pub fn new(active_qubit: usize, theta: f64) -> RotationZ {
        let gate_matrix = array![
            [
                Complex::<f64> {
                    re: (-theta / 2.).cos(),
                    im: (-theta / 2.).sin()
                },
                Complex::<f64> { re: 0., im: 0. }
            ],
            [
                Complex::<f64> { re: 0., im: 0. },
                Complex::<f64> {
                    re: (theta / 2.).cos(),
                    im: (theta / 2.).sin()
                }
            ]
        ];
        RotationZ {
            name: "RotationZ".to_string(),
            gate_matrix: gate_matrix,
            active_qubit: active_qubit,
            parameter: theta,
        }
    }
}
impl Gate for RotationZ {
    fn print(&self) {
        println!("RotationZ");
    }
    fn get_name(&self) -> &String {
        &self.name
    }
    fn get_matrix(&self) -> Array2<Complex<f64>> {
        self.gate_matrix.clone()
    }
    fn get_control_qubit(&self) -> Option<usize> {
        None
    }
    fn get_target_qubit(&self) -> usize {
        self.active_qubit
    }
    fn get_parameter(&self) -> Option<f64> {
        Some(self.parameter)
    }
    fn set_parameter(&mut self, parameter: f64) {
        *self = Self::new(self.active_qubit, parameter);
    }
}

/// struct S
///
/// # Attributes
/// * `name`: name of the gate
/// * `gate_matrix`: ArrayFire array representing the basic gate_matrix.
/// * `active_qubit`: Qubit id on which the S gate is applied
///
#[pyclass]
pub struct S {
    name: String,
    gate_matrix: Array2<Complex<f64>>,
    active_qubit: usize,
}
#[pymethods]
impl S {
    #[new]
    pub fn new(active_qubit: usize) -> S {
        let gate_matrix = array![
            [
                Complex::<f64> { re: 1., im: 0. },
                Complex::<f64> { re: 0., im: 0. }
            ],
            [
                Complex::<f64> { re: 0., im: 0. },
                Complex::<f64> { re: 0., im: 1. }
            ]
        ];
        S {
            name: "S".to_string(),
            gate_matrix: gate_matrix,
            active_qubit: active_qubit,
        }
    }
}
impl Gate for S {
    fn print(&self) {
        println!("S");
    }
    fn get_name(&self) -> &String {
        &self.name
    }
    fn get_matrix(&self) -> Array2<Complex<f64>> {
        self.gate_matrix.clone()
    }
    fn get_control_qubit(&self) -> Option<usize> {
        None
    }
    fn get_target_qubit(&self) -> usize {
        self.active_qubit
    }
    fn get_parameter(&self) -> Option<f64> {
        None
    }
    fn set_parameter(&mut self, _parameter: f64) {
        panic!("Cannot set parameter to S gate.");
    }
}

/// struct T
///
/// # Attributes
/// * `name`: name of the gate
/// * `gate_matrix`: ArrayFire array representing the basic gate_matrix.
/// * `active_qubit`: Qubit id on which the T gate is applied
///
#[pyclass]
pub struct T {
    name: String,
    gate_matrix: Array2<Complex<f64>>,
    active_qubit: usize,
}
#[pymethods]
impl T {
    #[new]
    pub fn new(active_qubit: usize) -> T {
        let gate_matrix = array![
            [
                Complex::<f64> { re: 1., im: 0. },
                Complex::<f64> { re: 0., im: 0. }
            ],
            [
                Complex::<f64> { re: 0., im: 0. },
                Complex::<f64> {
                    re: 2_f64.sqrt() / 2.,
                    im: 2_f64.sqrt() / 2.
                }
            ]
        ];
        T {
            name: "T".to_string(),
            gate_matrix: gate_matrix,
            active_qubit: active_qubit,
        }
    }
}
impl Gate for T {
    fn print(&self) {
        println!("T");
    }
    fn get_name(&self) -> &String {
        &self.name
    }
    fn get_matrix(&self) -> Array2<Complex<f64>> {
        self.gate_matrix.clone()
    }
    fn get_control_qubit(&self) -> Option<usize> {
        None
    }
    fn get_target_qubit(&self) -> usize {
        self.active_qubit
    }
    fn get_parameter(&self) -> Option<f64> {
        None
    }
    fn set_parameter(&mut self, _parameter: f64) {
        panic!("Cannot set parameter to T gate.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let _identity = Identity::new();
    }
    #[test]
    fn test_cnot() {
        let _cnot = CNOT::new(0, 1);
    }
    #[test]
    fn test_hadamard() {
        let _hadamard = Hadamard::new(0);
    }
    #[test]
    fn test_rotation_x() {
        let _rotation = RotationX::new(0, 0.5);
    }
    #[test]
    fn test_rotation_y() {
        let _rotation = RotationY::new(0, 0.5);
    }
    #[test]
    fn test_rotation_z() {
        let _rotation = RotationZ::new(0, 0.5);
    }
    #[test]
    fn test_pauli_x() {
        let _rotation = PauliX::new(0);
    }
    #[test]
    fn test_pauli_y() {
        let _rotation = PauliY::new(0);
    }
    #[test]
    fn test_pauli_z() {
        let _rotation = PauliZ::new(0);
    }
}
