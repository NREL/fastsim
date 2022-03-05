use pyo3::prelude::*;
extern crate ndarray;
use ndarray::{Array, Array1}; 

#[pyclass] 
#[derive(Debug, Clone)]
/// Struct containing time trace data 
pub struct Cycle{
    /// array of time [s]
    #[pyo3(get, set)]
    cycSecs: Array1<f64>,
    /// array of speed [m/s]
    #[pyo3(get, set)]
    cycMps: Array1<f64>,    
    /// array of grade [rise/run]
    #[pyo3(get, set)]
    cycGrade: Array1<f64>,
    /// array of max possible charge rate from roadway
    #[pyo3(get, set)]
    cycRoadType: Array1<f64>,
    #[pyo3(get, set)]
    name: String    
}


impl Cycle{
    
    fn get_cycMph(self){
        self.cycMps * params.mphPerMps
    }    
    fn set_cycMph(self, new_value):
        self.cycMps = new_value / params.mphPerMps

    fn get_time_s(self):
        return self.cycSecs

    fn set_time_s(self, new_value):
        self.cycSecs = new_value

    time_s = property(get_time_s, set_time_s)

    # time step deltas
    @property
    fn secs(self):
        return np.append(0.0, self.cycSecs[1:] - self.cycSecs[:-1]) 

    @property
    fn dt_s(self):
        return self.secs
    
    # distance at each time step
    @property
    fn cycDistMeters(self):
        return self.cycMps * self.secs

    @property
    fn delta_elev_m(self):
        """
        Cumulative elevation change w.r.t. to initial
        """
        return (self.cycDistMeters * self.cycGrade).cumsum()
    

}