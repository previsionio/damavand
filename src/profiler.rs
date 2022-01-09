
use std::time::Instant;
use mpi::topology::SystemCommunicator;
use mpi::traits::*;

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum OperationType {
    Forward,
    InterNodeCommunication,
    InterGPUCommunication,
    Sampling,
}

pub struct Profiler {
    pub iterations: usize,
    instant: Instant,
    cumulated_elapsed: f64
}

impl Profiler {
    pub fn new() -> Self{
        Self{
            iterations: 0,
            instant: Instant::now(),
            cumulated_elapsed: 0.,
        }
    }

    pub fn start(&mut self){
        SystemCommunicator::world().barrier();
        self.instant = Instant::now();
    }

    pub fn stop(&mut self){
        SystemCommunicator::world().barrier();
        self.cumulated_elapsed += self.instant.elapsed().as_millis() as f64;
        self.iterations += 1;
    }
    
    pub fn get_mean_elapsed(&self) -> f64{
        self.cumulated_elapsed / (self.iterations as f64)
    }
    
    pub fn reset(&mut self) {
        self.iterations = 0;
        self.cumulated_elapsed = 0.;
    }
}
