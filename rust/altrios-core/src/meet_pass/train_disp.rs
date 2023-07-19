use crate::train::LinkIdxTime;

use super::disp_imports::*;
mod free_path;

pub use free_path::FreePathStatus;

mod advance_rewind;

#[derive(Debug, Default, Clone, Serialize, Deserialize, SerdeAPI)]
pub struct TrainDisp {
    // Estimated time network. Does not change after creation
    est_times: Vec<EstTime>,
    est_time_statuses: Vec<EstTimeStatus>,

    // Path that the train takes through the network
    // (fixed path before disp_node_idx_free, free path after disp_node_idx_free)
    disp_path: Vec<DispNode>,
    // Optimized form of disp_path that has identical link_idxs (for arrive nodes, NA otherwise) plus an ending sentinel
    link_idx_path: Vec<LinkIdx>,
    // Optimized form of disp_path that is a fast set of all arrive node link_idxs
    links_on_path: IntSet<LinkIdx>,
    // All locations where the train diverged from the shortest path and why
    div_nodes: Vec<DivergeNode>,

    // Saved buffer for changing disp_path. Only modified within update_free_path
    disp_path_new: Vec<DispNode>,
    // Saved buffer for changing div_nodes. Only modified within update_free_path
    div_nodes_new: Vec<DivergeNode>,
    // Saved buffer of est_idxs currently blocked. Only modified within update_free_path
    est_idxs_blocked: Vec<EstIdx>,
    // Saved memory buffer of train_idxs blocked. Only modified within update_free_path
    train_idxs_blocking: Vec<TrainIdx>,

    // Time that the train will be updated at
    time_update: si::Time,
    time_update_next: si::Time,

    // Ordered list of all link indexes currently blocked by the train including lockouts
    link_idxs_blocking: Vec<LinkIdx>,
    // Location of boundary between fixed and free part of disp_path
    disp_node_idx_fixed: DispNodeIdx,
    disp_node_idx_free: DispNodeIdx,
    offset_fixed: si::Length,
    offset_free: si::Length,
    // Location of front of train along disp_path
    disp_node_idx_front: DispNodeIdx,
    // Location of back of train along disp_path
    disp_node_idx_back: DispNodeIdx,

    // Is the train currently blocked by another (same-direction) train?
    is_blocked: bool,

    //Const
    train_id: String,
    train_idx: TrainIdx,
    time_spacing: si::Time,
    dist_disp_path_search: si::Length,
    dist_fixed_max: si::Length,
    acc_startup: si::Acceleration,
}

impl TrainDisp {
    pub fn swap_link_idxs_blocking(&mut self, link_idxs: &mut Vec<LinkIdx>) {
        std::mem::swap(&mut self.link_idxs_blocking, link_idxs);
    }
    pub fn link_idxs_blocking(&self) -> &[LinkIdx] {
        &self.link_idxs_blocking
    }
    pub fn train_idx(&self) -> TrainIdx {
        self.train_idx
    }
    pub fn time_update(&self) -> si::Time {
        self.time_update
    }
    pub fn is_finished(&self) -> bool {
        self.disp_node_idx_free.idx() == self.disp_path.len()
    }
    pub fn is_blocked(&self) -> bool {
        self.is_blocked
    }
    pub fn fix_advance(&mut self) {
        assert!(self.time_update <= self.time_update_next);
        assert!(self.offset_fixed <= self.offset_free);
        assert!(self.disp_node_idx_fixed <= self.disp_node_idx_free);
        self.time_update = self.time_update_next;
        self.offset_fixed = self.offset_free;
        self.disp_node_idx_fixed = self.disp_node_idx_free;
    }
    pub fn calc_timed_path(&self) -> Vec<LinkIdxTime> {
        assert!(self.disp_node_idx_fixed.idx() == self.disp_path.len());
        let mut timed_path = Vec::with_capacity(self.disp_path.len() / 2);
        for disp_node in &self.disp_path {
            if disp_node.link_event.est_type == EstType::Arrive {
                timed_path.push(LinkIdxTime {
                    link_idx: disp_node.link_event.link_idx,
                    time: disp_node.time_pass,
                })
            }
        }
        timed_path
    }

    #[allow(clippy::too_many_arguments)]
    /// [TrainDisp] constructor method.
    pub fn new(
        train_id: String,
        train_idx: TrainIdx,
        time_depart: si::Time,
        time_spacing: si::Time,
        dist_disp_path_search: si::Length,
        dist_fixed_max: si::Length,
        acc_startup: si::Acceleration,
        est_time_net: EstTimeNet,
    ) -> Result<Self, anyhow::Error> {
        if train_idx.is_none() {
            bail!("Train disp cannot be created with train_idx=None!");
        }

        let est_times = est_time_net.val;

        // Initialize estimated time statuses
        let mut est_time_statuses = Vec::with_capacity(est_times.len());
        for est_time in &est_times {
            est_time_statuses.push(EstTimeStatus::new(est_time));
        }

        // Initialize disp_path with shortest path and update est_time_statuses
        let mut disp_path = Vec::with_capacity(est_times.len() / 2);
        let mut offset = si::Length::ZERO;
        let mut est_time_idx: EstIdx = 0;
        loop {
            let est_time = &est_times[est_time_idx.idx()];
            est_time_statuses[est_time_idx.idx()].is_on_path = true;
            disp_path.push(DispNode {
                offset,
                link_event: est_time.link_event,
                est_idx: est_time_idx,
                ..Default::default()
            });
            offset += est_time.dist_to_next;

            est_time_idx = est_time.idx_next;
            if est_time_idx == EST_IDX_NA {
                break;
            }
        }
        // Raise error if the dispatch path is too long
        u16::try_from(disp_path.len())?;

        // Initialize helper optimization objects for the dispatch path
        let disp_path_len_reserve = (disp_path.len() as f64 * 1.2).round() as usize;
        let mut link_idx_path = Vec::with_capacity(disp_path_len_reserve + 1);
        let mut links_on_path =
            IntSet::with_capacity_and_hasher(disp_path_len_reserve, Default::default());
        for disp_node in &disp_path {
            if disp_node.link_event.est_type == EstType::Arrive {
                links_on_path.insert(disp_node.link_event.link_idx);
                link_idx_path.push(disp_node.link_event.link_idx);
            } else {
                link_idx_path.push(track::LINK_IDX_NA);
            }
        }
        link_idx_path.push(track::LINK_IDX_NA); // Ending sentinel

        // Initialize diverge nodes
        let mut div_nodes = Vec::with_capacity(disp_path.len() / 5 + 2);
        div_nodes.push(Default::default()); // Starting sentinel
        div_nodes.push(DivergeNode {
            disp_node_idx: disp_path.len().try_from_idx().unwrap(),
            ..Default::default()
        }); // Ending sentinel

        let mut div_nodes_new = Vec::with_capacity(16.min(div_nodes.capacity()));
        div_nodes_new.push(Default::default()); // Starting sentinel

        // Initialize blocked info
        let mut train_idxs_blocking = Vec::with_capacity(64);
        train_idxs_blocking.push(Default::default()); // Extra value to ensure that base TrainIdxsView is unblocked

        Ok(Self {
            train_id,
            train_idx,
            time_spacing,
            dist_disp_path_search,
            dist_fixed_max,
            acc_startup,

            est_times,
            est_time_statuses,

            disp_path,
            link_idx_path,
            links_on_path,
            div_nodes,

            disp_path_new: Vec::with_capacity(32),
            div_nodes_new,
            est_idxs_blocked: Vec::with_capacity(64),
            train_idxs_blocking,

            time_update: time_depart,
            time_update_next: time_depart,

            link_idxs_blocking: Vec::with_capacity(16),
            offset_fixed: si::Length::ZERO,
            offset_free: si::Length::ZERO,
            disp_node_idx_fixed: None,
            disp_node_idx_free: None,
            disp_node_idx_front: None,
            disp_node_idx_back: None,
            is_blocked: false,
        })
    }
}

// TODO:  add dummy train
#[cfg(test)]
mod test_train_disp {
    use super::*;

    #[test]
    fn test_make_train_fwd() {
        let mut network_file_path = project_root::get_project_root().unwrap();
        network_file_path.push("../python/altrios/resources/networks/Taconite.yaml");
        let network =
            Vec::<Link>::from_file(network_file_path.as_os_str().to_str().unwrap()).unwrap();
        network.validate().unwrap();

        let speed_limit_train_sim = crate::train::speed_limit_train_sim_fwd();
        let est_times = make_est_times(&speed_limit_train_sim, &network).unwrap().0;
        TrainDisp::new(
            speed_limit_train_sim.train_id.clone(),
            NonZeroU16::new(1),
            speed_limit_train_sim.state.time,
            8.0 * uc::MIN,
            30.0 * uc::MI,
            10.0 * uc::MI,
            0.5 * uc::MPH / uc::S,
            est_times,
        )
        .unwrap();
    }

    #[test]
    fn test_make_train_rev() {
        // TODO: Make this test depend on a better file
        let mut network_file_path = project_root::get_project_root().unwrap();
        network_file_path.push("../python/altrios/resources/networks/Taconite.yaml");
        let network =
            Vec::<Link>::from_file(network_file_path.as_os_str().to_str().unwrap()).unwrap();
        network.validate().unwrap();
        let speed_limit_train_sim = crate::train::speed_limit_train_sim_rev();
        let est_times = make_est_times(&speed_limit_train_sim, &network).unwrap().0;
        TrainDisp::new(
            speed_limit_train_sim.train_id.clone(),
            NonZeroU16::new(1),
            speed_limit_train_sim.state.time,
            8.0 * uc::MIN,
            30.0 * uc::MI,
            10.0 * uc::MI,
            0.5 * uc::MPH / uc::S,
            est_times,
        )
        .unwrap();
    }
}
