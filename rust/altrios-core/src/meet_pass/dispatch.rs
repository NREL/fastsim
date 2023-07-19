use crate::combo_error::ComboErrors;
use crate::train::LinkIdxTime;

use super::disp_imports::*;
use super::train_disp::{FreePathStatus, TrainDisp};

#[readonly::make]
#[derive(Debug, PartialEq, Clone, Copy)]
struct TrainDispNext {
    pub time: si::Time,
    pub train_idx: TrainIdx,
}
impl TrainDispNext {
    pub fn new(time: si::Time, train_idx: TrainIdx) -> Self {
        assert!(!time.is_nan());
        assert!(train_idx.is_some());
        TrainDispNext { time, train_idx }
    }
    pub fn from_train_disp(train_disp: &TrainDisp) -> Self {
        Self::new(train_disp.time_update(), train_disp.train_idx())
    }
}

impl PartialOrd for TrainDispNext {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for TrainDispNext {}

impl Ord for TrainDispNext {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .time
            .partial_cmp(&self.time)
            .unwrap()
            .then_with(|| other.train_idx.cmp(&self.train_idx))
    }
}

/// Checks deadlock for all trains in the simulation after one train was moved.
/// Returns true if there is deadlock (at least one free path was not successfully modified), false otherwise
fn check_deadlock(
    train_disps: &mut [TrainDisp],
    links_blocked: &[TrainIdx],
    mut train_idx_begin: usize,
    train_idx_moved: TrainIdx,
    is_local: bool,
) -> Result<(bool, usize), ComboErrors<anyhow::Error>> {
    let mut has_deadlock = false;
    let mut errors = ComboErrors::<anyhow::Error>::new();

    // Swap the link idxs from the moved train. Necessary to avoid borrow checker issues.
    // This function must not return until swapping link idxs blocked back into the train!
    let mut link_idxs_blocked = vec![];
    train_disps[train_idx_moved.idx()].swap_link_idxs_blocking(&mut link_idxs_blocked);

    #[warn(clippy::mut_range_bound)]
    for (idx, train_disp) in train_disps.iter_mut().enumerate().skip(train_idx_begin) {
        if !train_disp.is_finished() {
            if idx != train_idx_moved.idx() {
                match train_disp.update_free_path(
                    train_idx_moved,
                    &link_idxs_blocked,
                    is_local,
                    links_blocked,
                ) {
                    Ok(path_status) => {
                        if let FreePathStatus::Blocked = path_status {
                            has_deadlock = true;
                        }
                    }
                    Err(error) => errors.push(error),
                }
            }
        } else if idx == train_idx_begin {
            train_idx_begin += 1;
        }
    }

    // Swap link idxs blocked back into train disp moved
    train_disps[train_idx_moved.idx()].swap_link_idxs_blocking(&mut link_idxs_blocked);

    if errors.len() > 0 {
        Err(errors)
    } else {
        Ok((has_deadlock, train_idx_begin))
    }
}

pub fn run_dispatch(
    network: &[Link],
    speed_limit_train_sims: &[SpeedLimitTrainSim],
    est_time_nets: Vec<EstTimeNet>,
    print_train_move: bool,
    print_train_exit: bool,
) -> anyhow::Result<Vec<Vec<LinkIdxTime>>> {
    let train_count = speed_limit_train_sims.len();
    if est_time_nets.len() != train_count {
        return Err(anyhow!(
            "Speed limit train sims length and est_time_nets length are unequal!"
        ));
    }

    let train_idx_width = (train_count as f64).log(10.0).floor() as usize + 1;
    let mut train_disps = Vec::with_capacity(train_count + 1);
    train_disps.push(TrainDisp::default());
    for (idx, (slts, est_time_net)) in speed_limit_train_sims
        .iter()
        .zip(est_time_nets.into_iter())
        .enumerate()
    {
        train_disps.push(TrainDisp::new(
            slts.train_id.clone(),
            (idx + 1).try_from_idx()?,
            slts.state.time,
            8.0 * uc::MIN,
            30.0 * uc::MI,
            10.0 * uc::MI,
            0.5 * uc::MPH / uc::S,
            est_time_net,
        )?);
    }

    let mut link_disp_auths = vec![
        vec![DispAuth {
            arrive_entry: f64::NEG_INFINITY * uc::S,
            arrive_exit: f64::NEG_INFINITY * uc::S,
            clear_entry: f64::NEG_INFINITY * uc::S,
            clear_exit: f64::NEG_INFINITY * uc::S,
            offset_front: f64::INFINITY * uc::M,
            offset_back: f64::INFINITY * uc::M,
            train_idx: None,
        }];
        network.len()
    ];
    let mut links_blocked = vec![None; network.len()];
    let mut train_idx_begin = 1usize;

    // Initialize train queue with all trains to be dispatched (skip first dummy idx)
    let mut train_disp_queue = BinaryHeap::with_capacity(train_disps.len());
    for (idx, train_disp) in train_disps.iter().enumerate().skip(1) {
        assert!(train_disp.train_idx().idx() == idx);
        train_disp_queue.push(TrainDispNext::from_train_disp(train_disp));
    }

    let mut train_idxs_blocked = Vec::with_capacity(train_disps.len() / 2);
    let mut has_deadlock = false;
    while !train_disp_queue.is_empty() {
        let train_idx_curr = train_disp_queue.pop().unwrap().train_idx;

        if print_train_move {
            println!(
                "Train {:0width$} was selected for advancing at time={:.2?} seconds",
                train_idx_curr.idx(),
                train_disps[train_idx_curr.idx()].time_update(),
                width = train_idx_width,
            );
        }

        // Advance one train until reaching a deadlock free configuration
        loop {
            if train_disps[train_idx_curr.idx()].advance(
                &mut link_disp_auths,
                &mut links_blocked,
                network,
            ) {
                (has_deadlock, train_idx_begin) = check_deadlock(
                    &mut train_disps,
                    &links_blocked,
                    train_idx_begin,
                    train_idx_curr,
                    true,
                )?;
                let train_curr = &mut train_disps[train_idx_curr.idx()];

                // If the train reaches the end of its path, break
                if train_curr.is_finished() {
                    assert!(
                        !has_deadlock,
                        "Train {:0width$} exited but there was deadlock!",
                        train_idx_curr.idx(),
                        width = train_idx_width,
                    );
                    train_curr.fix_advance();
                    if print_train_exit {
                        println!(
                            "Train {:0width$} exited at time={:.2?} seconds",
                            train_idx_curr.idx(),
                            train_curr.time_update(),
                            width = train_idx_width,
                        );
                    }
                    break;
                }

                // If there was deadlock and the train is blocked, rewind and break
                if has_deadlock && train_curr.is_blocked() {
                    train_curr.rewind(&mut link_disp_auths, &mut links_blocked, network);
                    (has_deadlock, train_idx_begin) = check_deadlock(
                        &mut train_disps,
                        &links_blocked,
                        train_idx_begin,
                        train_idx_curr,
                        false,
                    )?;
                    assert!(
                        !has_deadlock,
                        "Train {:0width$} was rewound to the last known good position but there was still deadlock!",
                        train_idx_curr.idx(),
                        width = train_idx_width,
                    );
                    break;
                }
            }

            // If there was not deadlock, fix any advance and break
            if !has_deadlock {
                train_disps[train_idx_curr.idx()].fix_advance();
                break;
            }
        }

        let train_curr = &train_disps[train_idx_curr.idx()];

        // If the train is blocked and not finished, add it to the blocked trains
        if train_curr.is_blocked() && !train_curr.is_finished() {
            train_idxs_blocked.push(train_idx_curr);
        }
        // Otherwise, add it and all currently blocked trains back to the queue
        else {
            if !train_curr.is_finished() {
                train_disp_queue.push(TrainDispNext {
                    time: train_curr.time_update(),
                    train_idx: train_idx_curr,
                });
            }
            train_idxs_blocked.drain(..).for_each(|train_idx| {
                train_disp_queue.push(TrainDispNext {
                    time: train_disps[train_idx.idx()].time_update(),
                    train_idx,
                });
                debug_assert!(train_idx != train_idx_curr);
            });
        }
    }
    if !train_idxs_blocked.is_empty() {
        bail!("The following trains got stuck! {:?}", train_idxs_blocked);
    }

    Ok(train_disps[1..]
        .iter()
        .map(|x| x.calc_timed_path())
        .collect::<Vec<Vec<LinkIdxTime>>>())
}

#[cfg_attr(feature = "pyo3", pyfunction(name = "run_dispatch"))]
pub fn run_dispatch_py(
    network: Vec<Link>,
    speed_limit_train_sims: crate::train::SpeedLimitTrainSimVec,
    est_time_vec: Vec<EstTimeNet>,
    print_train_move: bool,
    print_train_exit: bool,
) -> anyhow::Result<Vec<Vec<LinkIdxTime>>> {
    run_dispatch(
        &network,
        &speed_limit_train_sims.0,
        est_time_vec,
        print_train_move,
        print_train_exit,
    )
}

#[cfg(test)]
mod test_dispatch {
    use super::*;
    // use crate::testing::*;

    #[test]
    fn test_empty_dispatch() {
        let output = run_dispatch(&[], &[], vec![], false, false).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_simple_dispatch() {
        let mut network_file_path = project_root::get_project_root().unwrap();
        network_file_path.push("../python/altrios/resources/networks/Taconite.yaml");
        let network =
            Vec::<Link>::from_file(network_file_path.as_os_str().to_str().unwrap()).unwrap();
        network.validate().unwrap();

        let train_sims = vec![
            crate::train::speed_limit_train_sim_fwd(),
            crate::train::speed_limit_train_sim_rev(),
        ];
        // &vec![
        //     crate::train::speed_limit_train_sim_fwd(),
        //     crate::train::speed_limit_train_sim_rev(),
        //     crate::train::speed_limit_train_sim_fwd(),
        //     crate::train::speed_limit_train_sim_rev(),
        //     crate::train::speed_limit_train_sim_fwd(),
        //     crate::train::speed_limit_train_sim_rev(),
        //     crate::train::speed_limit_train_sim_fwd(),
        //     crate::train::speed_limit_train_sim_rev()
        // ],

        let est_time_vec = train_sims
            .iter()
            .map(|slts| make_est_times(slts, &network).unwrap().0)
            .collect::<Vec<EstTimeNet>>();
        let _output = run_dispatch(&network, &train_sims, est_time_vec, true, true).unwrap();
    }
}
