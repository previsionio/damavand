
use crate::qubit_backend::circuit::Circuit;
use crate::profiler::OperationType;

impl Circuit{

    pub fn start_profiling_forward(&mut self) {
        self.profilers.get_mut(&OperationType::Forward).unwrap().start();
    }

    pub fn stop_profiling_forward(&mut self) {
        self.profilers.get_mut(&OperationType::Forward).unwrap().stop();
    }

    pub fn get_mean_elapsed_forward(&self) -> f64{
        self.profilers.get(&OperationType::Forward).unwrap().get_mean_elapsed()
    }

    pub fn start_profiling_inter_node_communications(&mut self) {
        self.profilers.get_mut(&OperationType::InterNodeCommunication).unwrap().start();
    }

    pub fn stop_profiling_inter_node_communications(&mut self) {
        self.profilers.get_mut(&OperationType::InterNodeCommunication).unwrap().stop();
    }

    pub fn get_mean_elapsed_inter_node_communications(&self) -> f64{
        self.profilers.get(&OperationType::InterNodeCommunication).unwrap().get_mean_elapsed()
    }

    pub fn start_profiling_inter_gpu_communications(&mut self) {
        self.profilers.get_mut(&OperationType::InterGPUCommunication).unwrap().start();
    }

    pub fn stop_profiling_inter_gpu_communications(&mut self) {
        self.profilers.get_mut(&OperationType::InterGPUCommunication).unwrap().stop();
    }

    pub fn get_mean_elapsed_inter_gpu_communications(&self) -> f64{
        self.profilers.get(&OperationType::InterGPUCommunication).unwrap().get_mean_elapsed()
    }

    pub fn start_profiling_sampling(&mut self) {
        self.profilers.get_mut(&OperationType::Sampling).unwrap().start();
    }

    pub fn stop_profiling_sampling(&mut self) {
        self.profilers.get_mut(&OperationType::Sampling).unwrap().stop();
    }

    pub fn get_mean_elapsed_sampling(&self) -> f64{
        self.profilers.get(&OperationType::Sampling).unwrap().get_mean_elapsed()
    }
}
