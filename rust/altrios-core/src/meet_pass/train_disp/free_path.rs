use super::super::disp_imports::*;
use super::TrainDisp;

enum LinkOptType {
    None,
    Single(LinkIdx),
    Range(usize, usize),
    Check,
}

impl LinkOptType {
    pub fn new(link_idxs_blocking: &[LinkIdx], link_idxs_on_path: &IntSet<LinkIdx>) -> Self {
        let mut link_opt_type = LinkOptType::None;
        for link_idx in link_idxs_blocking {
            if link_idxs_on_path.contains(link_idx) {
                link_opt_type = match link_opt_type {
                    LinkOptType::None => LinkOptType::Single(*link_idx),
                    LinkOptType::Single(link_idx_prev) => LinkOptType::Range(
                        link_idx.idx().min(link_idx_prev.idx()),
                        link_idx.idx().max(link_idx_prev.idx()),
                    ),
                    LinkOptType::Range(link_idx_min, link_idx_max) => LinkOptType::Range(
                        link_idx.idx().min(link_idx_min),
                        link_idx.idx().max(link_idx_max),
                    ),
                    _ => unreachable!("Impossible link_opt_type value!"),
                };
            }
        }

        // If the type is check and the difference is sufficiently small, set the type to range
        if let LinkOptType::Range(link_idx_min, link_idx_max) = link_opt_type {
            let link_idx_diff = link_idx_max - link_idx_min;
            if link_idx_diff <= 16 {
                LinkOptType::Range(link_idx_min, link_idx_diff)
            } else {
                LinkOptType::Check
            }
        } else {
            link_opt_type
        }
    }
}

fn calc_idx_sentinels(
    mut div_idx: usize,
    train_idx_sentinel: TrainIdx,
    div_nodes: &[DivergeNode],
) -> (usize, usize) {
    assert!(div_idx < div_nodes.len());
    assert!(div_nodes.last().unwrap().train_idx == train_idx_sentinel);
    // SAFETY: div_idx starts within div_nodes and cannot pass the ending sentinel
    let disp_node_idx_sentinel = unsafe {
        // Find the first matching train idx
        while div_nodes.get_unchecked(div_idx).train_idx != train_idx_sentinel {
            div_idx += 1;
        }
        div_nodes.get_unchecked(div_idx).disp_node_idx
    };

    // Find the first diverge node with the next disp node idx to mark the split point
    div_idx += 1;
    if div_idx < div_nodes.len() {
        while div_nodes[div_idx].disp_node_idx == disp_node_idx_sentinel {
            div_idx += 1;
        }
    }
    (disp_node_idx_sentinel.idx(), div_idx)
}

fn find_train_intersect(
    mut idx_split: usize,
    idx_sentinel: usize,
    link_opt_type: &LinkOptType,
    link_idx_path: &mut [LinkIdx],
    links_blocked: &[TrainIdx],
) -> usize {
    // If the current train definitely reaches the straighten point first, do not search
    if idx_split >= idx_sentinel {
        return idx_split;
    }

    assert!(idx_sentinel < link_idx_path.len());
    match link_opt_type {
        LinkOptType::Single(link_idx_check) => {
            // SAFETY: Starts in range because idx_split < idx_sentinel and idx_sentinel < link_idx_path.len()
            // Stays in range because self.link_idx_path[idx_sentinel] = link_idx_check
            unsafe {
                // Save correct value and set the sentinel value
                let link_idx_save = *link_idx_path.get_unchecked(idx_sentinel);
                *link_idx_path.get_unchecked_mut(idx_sentinel) = *link_idx_check;

                // Advance until finding link_idx_check
                while link_idx_path.get_unchecked(idx_split) != link_idx_check {
                    idx_split += 1;
                }

                // Undo the sentinel overwrite
                *link_idx_path.get_unchecked_mut(idx_sentinel) = link_idx_save;
            }
        }
        LinkOptType::Range(link_idx_min, link_idx_diff) => {
            // SAFETY: Starts in range because idx_split < idx_sentinel and idx_sentinel < link_idx_path.len()
            // Stays in range because self.link_idx_path[idx_sentinel] = link_idx_check
            // Continues to stay in range because idx_split < idx_sentinel
            unsafe {
                // Save correct value and set the sentinel value
                let link_idx_save = *link_idx_path.get_unchecked(idx_sentinel);
                *link_idx_path.get_unchecked_mut(idx_sentinel) = LinkIdx::new(*link_idx_min as u32);

                // Advance until finding an intersection or the sentinel
                loop {
                    while link_idx_path
                        .get_unchecked(idx_split)
                        .idx()
                        .wrapping_sub(*link_idx_min)
                        > *link_idx_diff
                    {
                        idx_split += 1;
                    }
                    // Exit if straighten point is reached or train intersection is found
                    if idx_split == idx_sentinel
                        || links_blocked[link_idx_path.get_unchecked(idx_split).idx()].is_some()
                    {
                        break;
                    }

                    idx_split += 1;
                }

                // Undo the sentinel overwrite
                *link_idx_path.get_unchecked_mut(idx_sentinel) = link_idx_save;
            }
        }
        LinkOptType::Check => {
            // SAFETY: Starts in range because idx_split < idx_sentinel and idx_sentinel < link_idx_path.len()
            unsafe {
                while idx_split < idx_sentinel {
                    if links_blocked[link_idx_path.get_unchecked(idx_split).idx()].is_some() {
                        break;
                    }
                    idx_split += 1;
                }
            }
        }
        _ => unreachable!("Impossible link_opt_type value!"),
    }
    idx_split
}

/// Add all non-duplicate blocking trains from the add view to the base view.
/// Requires that the base view be positioned at the end of idxs_blocking.
fn add_blocking_trains(
    trains_blocking: &mut Vec<TrainIdx>,
    trains_view_base: &TrainIdxsView,
    trains_view_add: &TrainIdxsView,
) -> TrainIdxsView {
    assert!(trains_view_base.idx_begin <= trains_view_base.idx_end);
    assert!(trains_blocking.len() == trains_view_base.idx_end.idx());
    trains_blocking.reserve(trains_view_add.len() + 1);

    // Add space for sentinel
    trains_blocking.push(None);
    for idx_add in trains_view_add.idx_begin..trains_view_add.idx_end {
        let train_add = trains_blocking[idx_add.idx()];
        let mut idx_test = trains_view_base.idx_begin.idx();

        // SAFETY: idx_test = trains_view_base.idx_begin <= trains_view_base.idx_end
        // and trains_blocking[trains_view_base.idx_end] = train_add (sentinel)
        unsafe {
            *trains_blocking.get_unchecked_mut(trains_view_base.idx_end.idx()) = train_add;
            while *trains_blocking.get_unchecked(idx_test) != train_add {
                idx_test += 1;
            }
        }

        // Add the train if the sentinel was reached (train not found in dedup search)
        if idx_test == trains_view_base.idx_end.idx() {
            trains_blocking.push(train_add);
        }
    }

    // Pop last element (may be sentinel or real train)
    let train_save = trains_blocking.pop().unwrap();
    // If it was a real train, overwrite the sentinel to create a continuous range
    if trains_view_base.idx_end.idx() < trains_blocking.len() {
        trains_blocking[trains_view_base.idx_end.idx()] = train_save;
    }

    TrainIdxsView {
        idx_begin: trains_view_base.idx_begin,
        idx_end: trains_blocking.len().try_into().unwrap(),
    }
}

/// Add all non-duplicate blocking trains from both the large and small views
fn add_all_blocking_trains(
    trains_blocking: &mut Vec<TrainIdx>,
    trains_view_large: &TrainIdxsView,
    trains_view_small: &TrainIdxsView,
) -> TrainIdxsView {
    trains_blocking.reserve(trains_view_large.len() + trains_view_small.len() + 1);
    trains_blocking.extend_from_within(trains_view_large.range());
    add_blocking_trains(
        trains_blocking,
        &TrainIdxsView {
            idx_begin: (trains_blocking.len() - trains_view_large.len())
                .try_into()
                .unwrap(),
            idx_end: trains_blocking.len().try_into().unwrap(),
        },
        trains_view_small,
    )
}

fn concat_train_idx_views(
    trains_blocking: &mut Vec<TrainIdx>,
    trains_view: &TrainIdxsView,
    trains_view_add: &TrainIdxsView,
) -> TrainIdxsView {
    if trains_view_add.is_empty()
        || (trains_view.idx_begin <= trains_view_add.idx_begin
            && trains_view_add.idx_end <= trains_view.idx_end)
    {
        *trains_view
    } else if trains_view.is_empty()
        || (trains_view_add.idx_begin <= trains_view.idx_begin
            && trains_view.idx_end <= trains_view_add.idx_end)
    {
        *trains_view_add
    } else if trains_blocking.len() == trains_view.idx_end.idx() {
        add_blocking_trains(trains_blocking, trains_view, trains_view_add)
    } else if trains_blocking.len() == trains_view_add.idx_end.idx() {
        add_blocking_trains(trains_blocking, trains_view_add, trains_view)
    } else if trains_view.len() >= trains_view_add.len() {
        add_all_blocking_trains(trains_blocking, trains_view, trains_view_add)
    } else {
        add_all_blocking_trains(trains_blocking, trains_view_add, trains_view)
    }
}

/// Return status from update_free_path function
pub enum FreePathStatus {
    /// Free path is valid
    UpdateSuccess,
    /// Free path update failed due to opposite direction train
    Blocked,
}

impl TrainDisp {
    /// Update this train's free path to navigate around the train_idx_moved
    pub fn update_free_path(
        &mut self,
        train_idx_moved: TrainIdx,
        link_idxs_blocking: &[LinkIdx],
        is_local: bool,
        links_blocked: &[TrainIdx],
    ) -> anyhow::Result<FreePathStatus> {
        if self.disp_path.len() <= self.disp_node_idx_free.idx() {
            bail!(
                "Update free path cannot be called on train {} after it has finished its trip!",
                self.train_idx.idx()
            );
        }

        // This does NOT need to be refreshed when link_idxs_on_path changes because the changed path does not intersect with moved train
        let link_opt_type = LinkOptType::new(link_idxs_blocking, &self.links_on_path);

        // Smallest offset allowed to be considered when walking backwards
        let offset_lowest =
            self.disp_path[self.disp_node_idx_free.idx()].offset - self.dist_fixed_max;

        let mut idx_split = 1;
        let mut div_idx_split = 1;
        self.div_nodes.last_mut().unwrap().train_idx = train_idx_moved; // Adjust ending sentinel

        // Walk entire dispatch path, adjusting free path when necessary
        while idx_split < self.disp_path.len() {
            let (idx_sentinel, div_idx_sentinel) =
                calc_idx_sentinels(div_idx_split, train_idx_moved, &self.div_nodes);

            // If the current train cannot intersect with the moved train, do not search along the dispatch path
            if match link_opt_type {
                LinkOptType::None => true,
                LinkOptType::Single(link_idx) => !self.links_on_path.contains(&link_idx),
                _ => false,
            } {
                idx_split = idx_sentinel;
            }
            // Otherwise, search along the dispatch path and find the first intersection
            else {
                assert!(idx_split <= idx_sentinel);
                if idx_split < self.disp_node_idx_free.idx() {
                    idx_split = self.disp_node_idx_free.idx().min(idx_sentinel);
                }
                assert!(idx_split < self.disp_path.len());

                idx_split = find_train_intersect(
                    idx_split,
                    idx_sentinel,
                    &link_opt_type,
                    &mut self.link_idx_path,
                    links_blocked,
                );
            }

            // If the split point has reached the end of the path, break
            if self.disp_path.len() <= idx_split {
                break;
            }

            assert!(self.train_idxs_blocking.len() == 1);
            assert!(self.disp_path_new.is_empty());

            // Limit the maximum search distance to be more reasonable
            let offset_cancel = if is_local {
                offset_lowest.max(self.disp_path[idx_split].offset - self.dist_disp_path_search)
            } else {
                offset_lowest
            };

            // If the split point is the sentinel, the diverge split point is also the sentinel
            if idx_split == idx_sentinel {
                div_idx_split = div_idx_sentinel;
                assert!(self.div_nodes[div_idx_split - 1].disp_node_idx.idx() == idx_split);
            } else {
                while self.div_nodes[div_idx_split].disp_node_idx.idx() <= idx_split {
                    div_idx_split += 1;
                }
            }
            assert!(self.div_nodes[div_idx_split - 1].disp_node_idx.idx() <= idx_split);
            assert!(self.div_nodes[div_idx_split].disp_node_idx.idx() > idx_split);

            let mut idx_save = idx_split;
            if self.disp_node_idx_fixed.idx() <= idx_split {
                idx_save = self.disp_node_idx_fixed.idx();
            } else {
                while self.disp_path[idx_save].link_event.est_type == EstType::Fake {
                    idx_save += 1;
                }
            }

            let idx_split_save = idx_split;
            // Search for a new path
            loop {
                let idx_join_new = idx_split + self.disp_path_new.len();
                // Current disp_node and est_time_node when searching
                let disp_node_curr = if let Some(disp_node_curr) = self.disp_path_new.last() {
                    disp_node_curr
                } else {
                    &self.disp_path[idx_split - 1]
                };
                let est_curr = &self.est_times[disp_node_curr.est_idx.idx()];

                assert!(!self.est_time_statuses[disp_node_curr.est_idx.idx()].is_blocked());
                assert!(est_curr.idx_next != EST_IDX_NA);
                assert!(
                    self.div_nodes[div_idx_split - 1].disp_node_idx
                        < self.div_nodes[div_idx_split].disp_node_idx
                );

                // Block next est time node with previously stored diverging trains if applicable
                if self.div_nodes[div_idx_split - 1].disp_node_idx.idx() == idx_join_new {
                    assert!(est_curr.idx_next_alt != EST_IDX_NA);
                    assert!(self.est_time_statuses[est_curr.idx_next_alt.idx()].is_on_path);
                    assert!(!self.est_time_statuses[est_curr.idx_next.idx()].is_blocked());

                    let mut div_idx_save = div_idx_split;
                    while self.div_nodes[div_idx_split - 1].disp_node_idx.idx() == idx_join_new {
                        if self.div_nodes[div_idx_split - 1].train_idx == train_idx_moved {
                            div_idx_save = 0;
                        }
                        div_idx_split -= 1;
                    }

                    // Block next est time node immediately if train idx moved is not in the div nodes
                    if div_idx_save > 0 {
                        assert!(self.est_time_statuses[est_curr.idx_next_alt.idx()].is_blocked());
                        self.est_idxs_blocked.push(est_curr.idx_next);
                        self.est_time_statuses[est_curr.idx_next.idx()].train_idxs_view =
                            TrainIdxsView {
                                idx_begin: self.train_idxs_blocking.len().try_into().unwrap(),
                                idx_end: (self.train_idxs_blocking.len() + div_idx_save
                                    - div_idx_split)
                                    .try_into()
                                    .unwrap(),
                            };
                        for div_node in &self.div_nodes[div_idx_split..div_idx_save] {
                            self.train_idxs_blocking.push(div_node.train_idx);
                        }
                    }
                }
                assert!(self.div_nodes[div_idx_split - 1].disp_node_idx.idx() < idx_join_new);

                // If the train is evaluating the free path and the next node is an arrive node
                let est_status_next = &self.est_time_statuses[est_curr.idx_next.idx()];
                if !est_status_next.is_blocked()
                    && self.disp_node_idx_free.idx() <= idx_join_new
                    && est_status_next.est_type == EstType::Arrive
                {
                    // If the next node is occupied by a train, mark it as blocked
                    if let Some(train_idx) = links_blocked[est_status_next.link_idx.idx()] {
                        self.est_idxs_blocked.push(est_curr.idx_next);
                        self.est_time_statuses[est_curr.idx_next.idx()].train_idxs_view =
                            TrainIdxsView {
                                idx_begin: self.train_idxs_blocking.len().try_into().unwrap(),
                                idx_end: (self.train_idxs_blocking.len() + 1).try_into().unwrap(),
                            };
                        self.train_idxs_blocking.push(Some(train_idx));
                    }
                }

                // Move one node along disp path
                let est_status_next = &self.est_time_statuses[est_curr.idx_next.idx()];
                let est_status_next_alt = &self.est_time_statuses[est_curr.idx_next_alt.idx()];
                // If the next est time is blocked
                if est_status_next.is_blocked() {
                    // Rewind if the alternate est time is also blocked
                    if est_curr.idx_next_alt == EST_IDX_NA || est_status_next_alt.is_blocked() {
                        // Shift the save node if applicable
                        if idx_join_new <= self.disp_node_idx_free.idx()
                            && disp_node_curr.link_event.est_type != EstType::Fake
                        {
                            idx_save -= 1;
                            while self.disp_path[idx_save].link_event.est_type == EstType::Fake {
                                idx_save -= 1;
                            }

                            assert!(idx_save < self.disp_node_idx_fixed.idx());
                            assert!(
                                disp_node_curr.link_event == self.disp_path[idx_save].link_event
                            );
                        }

                        // Add blocking train idxs to the current node
                        self.est_idxs_blocked.push(disp_node_curr.est_idx);
                        self.est_time_statuses[disp_node_curr.est_idx.idx()].train_idxs_view =
                            if est_curr.idx_next_alt == EST_IDX_NA {
                                est_status_next.train_idxs_view
                            } else {
                                concat_train_idx_views(
                                    &mut self.train_idxs_blocking,
                                    &est_status_next.train_idxs_view,
                                    &est_status_next_alt.train_idxs_view,
                                )
                            };

                        // Rewind through new diverge nodes if needed
                        while self.div_nodes_new.last().unwrap().disp_node_idx.idx() == idx_join_new
                        {
                            self.div_nodes_new.pop();
                        }
                        assert!(
                            self.div_nodes_new.last().unwrap().disp_node_idx.idx() < idx_join_new
                        );

                        // Try to rewind by one dispatch node
                        if !self.disp_path_new.is_empty() {
                            self.disp_path_new.pop();
                        } else if offset_cancel < disp_node_curr.offset && 1 < idx_split {
                            idx_split -= 1;
                        }
                        // If rewinding is not possible, stop and return failed
                        else {
                            assert!(self.disp_path_new.is_empty());
                            assert!(self.div_nodes_new.len() == 1);
                            assert!(idx_join_new == idx_split);

                            self.disp_node_idx_free = self.disp_node_idx_fixed;
                            self.reset_blocking();
                            return Ok(FreePathStatus::Blocked);
                        }
                    }
                    // Deposit blocking trains and go on the alternate if the next est time is blocked
                    // and the alternate is not blocked and is not on the old path
                    else if !est_status_next_alt.is_on_path {
                        if idx_join_new <= self.disp_node_idx_free.idx() {
                            self.disp_node_idx_free =
                                (self.disp_node_idx_free.idx() + 1).try_from_idx().unwrap();
                        }

                        // Deposit all blocking trains
                        self.div_nodes_new
                            .reserve(est_status_next.train_idxs_view.len());
                        for train_idx in
                            &self.train_idxs_blocking[est_status_next.train_idxs_view.range()]
                        {
                            self.div_nodes_new.push(DivergeNode {
                                train_idx: *train_idx,
                                disp_node_idx: idx_join_new.try_from_idx().unwrap(),
                            });
                        }

                        self.disp_path_new.push(DispNode {
                            offset: disp_node_curr.offset,
                            time_pass: disp_node_curr.time_pass,
                            est_idx: est_curr.idx_next_alt,
                            ..Default::default()
                        });
                    }
                    // The old path has been found at a split node, so only blocking trains are changed
                    else {
                        assert!(self.disp_path_new.is_empty());
                        assert!(self.div_nodes_new.len() == 1);
                        assert!(idx_join_new == idx_split);
                        assert!(est_curr.idx_next_alt == self.disp_path[idx_split].est_idx);
                        assert!(idx_split == self.div_nodes[div_idx_split].disp_node_idx.idx());

                        // Calculate the base range
                        let div_idx_base = div_idx_split;
                        let disp_node_idx = idx_split.try_from_idx().unwrap();
                        assert!(disp_node_idx == self.div_nodes[div_idx_split].disp_node_idx);
                        div_idx_split += 1;
                        while disp_node_idx == self.div_nodes[div_idx_split].disp_node_idx {
                            div_idx_split += 1;
                        }

                        // Splice in the new diverge nodes
                        let divs_add = self.train_idxs_blocking
                            [est_status_next.train_idxs_view.range()]
                        .iter()
                        .map(|x| DivergeNode {
                            train_idx: *x,
                            disp_node_idx,
                        });
                        self.div_nodes.splice(div_idx_base..div_idx_split, divs_add);
                        div_idx_split = div_idx_base + est_status_next.train_idxs_view.len();

                        self.disp_node_idx_free = self.disp_node_idx_fixed;
                        self.reset_blocking();
                        break;
                    }
                }
                // If the next est time is not blocked and is not on the old path, advance to it
                else if !est_status_next.is_on_path {
                    // If the saved disp path must be used
                    if idx_save < self.disp_node_idx_fixed.idx() {
                        self.disp_node_idx_free = (self.disp_node_idx_fixed.idx() - idx_save
                            + idx_join_new)
                            .try_from_idx()
                            .unwrap();
                        // Add node if it is a space match
                        if est_status_next.link_event() == self.disp_path[idx_save].link_event {
                            assert!(utils::almost_eq_uom(
                                &self.disp_path[idx_save].offset,
                                &(disp_node_curr.offset + est_curr.dist_to_next),
                                None
                            ));
                            self.disp_path_new.push(self.disp_path[idx_save]);
                            self.disp_path_new.last_mut().unwrap().est_idx = est_curr.idx_next;
                            idx_save += 1;
                            while self.disp_path[idx_save].link_event.est_type == EstType::Fake {
                                idx_save += 1;
                            }
                        }
                        // Otherwise, block the path with no trains
                        else {
                            self.est_idxs_blocked.push(est_curr.idx_next);
                            self.est_time_statuses[est_curr.idx_next.idx()].block_empty();
                        }
                    }
                    // Otherwise, add the next node to the new disp path
                    else {
                        self.disp_node_idx_free = self
                            .disp_node_idx_free
                            .min(idx_join_new.try_from_idx().unwrap());
                        self.disp_path_new.push(DispNode {
                            offset: disp_node_curr.offset + est_curr.dist_to_next,
                            link_event: est_status_next.link_event(),
                            est_idx: est_curr.idx_next,
                            ..Default::default()
                        })
                    }
                }
                // If the next est time is not blocked and is on the old path, join to the old path
                else {
                    // Find the join point on the original path
                    let mut idx_join_base = idx_split_save;
                    while est_curr.idx_next != self.disp_path[idx_join_base].est_idx {
                        idx_join_base += 1;
                    }
                    let idx_join_base = idx_join_base;
                    let _: DispNodeIdx = (self.disp_path.len() - idx_join_base + idx_join_new)
                        .try_from_idx()
                        .unwrap();

                    // Update est_time_statuses, link_idxs_on_path, offsets,
                    // disp_node_idx_front, and disp_node_idx_back.
                    self.update_free_path_helpers(
                        idx_split,
                        idx_join_base,
                        disp_node_curr.offset + est_curr.dist_to_next
                            - self.disp_path[idx_join_base].offset,
                    );

                    // Splice in new diverge nodes
                    div_idx_split =
                        self.update_div_nodes(div_idx_split, idx_join_base, idx_join_new);

                    // Splice in new disp nodes
                    self.link_idx_path.splice(
                        idx_split..idx_join_base,
                        self.disp_path_new.iter().map(|x| {
                            if x.link_event.est_type == EstType::Arrive {
                                x.link_event.link_idx
                            } else {
                                track::LINK_IDX_NA
                            }
                        }),
                    );
                    self.disp_path
                        .splice(idx_split..idx_join_base, self.disp_path_new.drain(..));
                    idx_split = idx_join_new;

                    // Verify that disp_node_idx_free is correct
                    assert!(
                        self.disp_path[self.disp_node_idx_free.idx()].time_pass
                            == uc::S * f64::INFINITY
                    );
                    if self.disp_node_idx_free.is_some()
                        && self.disp_path[self.disp_node_idx_free.idx() - 1].offset
                            != si::Length::ZERO
                    {
                        assert!(
                            self.disp_path[self.disp_node_idx_free.idx() - 1].time_pass
                                != uc::S * f64::INFINITY
                        );
                    }
                    self.disp_node_idx_fixed = self.disp_node_idx_free;

                    break;
                }
            }
        }

        self.validate_free_path(links_blocked)?;
        Ok(FreePathStatus::UpdateSuccess)
    }

    /// Finds where to splice in new diverge nodes and performs the splice.
    /// Returns the updated value of div_idx_split
    fn update_div_nodes(
        &mut self,
        div_idx_split: usize,
        idx_join_base: usize,
        idx_join_new: usize,
    ) -> usize {
        // Find the div node join point on the original path
        let mut div_idx_join_base = div_idx_split;
        while self.div_nodes[div_idx_join_base].disp_node_idx.idx() <= idx_join_base {
            div_idx_join_base += 1;
        }
        let div_idx_join_base = div_idx_join_base;
        let div_idx_join_new = div_idx_split + self.div_nodes_new.len() - 1;
        let _: DispNodeIdx = (self.div_nodes.len() - div_idx_join_base + div_idx_join_new)
            .try_from_idx()
            .unwrap();

        // Adjust saved disp node idx values
        if idx_join_base != idx_join_new {
            for div_node in &mut self.div_nodes[div_idx_join_base..] {
                div_node.disp_node_idx = (div_node.disp_node_idx.idx() + idx_join_new
                    - idx_join_base)
                    .try_from_idx()
                    .unwrap();
            }
        }

        // Splice in new div nodes
        self.div_nodes.splice(
            div_idx_split..div_idx_join_base,
            self.div_nodes_new.drain(1..),
        );
        div_idx_join_new
    }

    /// Unblock est_time_statuses, reset est_idxs_blocking,
    /// and reset train_idxs_blocking.
    fn reset_blocking(&mut self) {
        for est_idx in &self.est_idxs_blocked {
            self.est_time_statuses[est_idx.idx()].unblock();
        }
        self.est_idxs_blocked.clear();
        self.train_idxs_blocking.truncate(1);
    }

    /// Update est_time_statuses and link_idxs_on_path
    /// and update offsets for last part of original path
    /// and update the front and back dispatch nodes.
    fn update_free_path_helpers(
        &mut self,
        idx_split: usize,
        idx_join_base: usize,
        offset_change: si::Length,
    ) {
        self.reset_blocking();

        // Remove old path from est_time_statuses and links_on_path
        for disp_node in &self.disp_path[idx_split..idx_join_base] {
            self.est_time_statuses[disp_node.est_idx.idx()].is_on_path = false;
            if disp_node.link_event.est_type == EstType::Arrive {
                assert!(self.links_on_path.remove(&disp_node.link_event.link_idx));
            }
        }
        // Add new path to est_time_statuses and links_on_path
        for disp_node in &self.disp_path_new {
            self.est_time_statuses[disp_node.est_idx.idx()].is_on_path = true;
            if disp_node.link_event.est_type == EstType::Arrive {
                assert!(self.links_on_path.insert(disp_node.link_event.link_idx));
            }
        }

        // Update offset values
        if !utils::almost_eq_uom(&offset_change, &si::Length::ZERO, None) {
            self.disp_path[idx_join_base..]
                .iter_mut()
                .for_each(|x| x.offset += offset_change);
        }

        // Update front and back dispatch nodes
        let (idx_min, idx_max) = if self.disp_node_idx_back < self.disp_node_idx_front {
            (&mut self.disp_node_idx_front, &mut self.disp_node_idx_back)
        } else {
            (&mut self.disp_node_idx_back, &mut self.disp_node_idx_front)
        };
        assert!(idx_max.idx() < idx_join_base);

        // If the train has passed the split point
        if idx_split < idx_max.idx() {
            let mut idx_new = 0;
            let mut update_disp_node_idx = |idx_update: &mut DispNodeIdx| {
                // This exits because some place on the new path must match
                while self.disp_path_new[idx_new].link_event
                    != self.disp_path[idx_update.idx()].link_event
                {
                    idx_new += 1;
                }
                *idx_update = (idx_split + idx_new).try_from_idx().unwrap();
            };

            if idx_split < idx_min.idx() {
                update_disp_node_idx(idx_min);
            }
            update_disp_node_idx(idx_max);
        }
    }

    fn validate_free_path(&self, links_blocked: &[TrainIdx]) -> ValidationResults {
        let mut errors = ValidationErrors::new();

        // Ensure that the offsets and estimated time linking is valid
        for (disp_node_curr, disp_node_next) in
            self.disp_path.windows(2).map(|vals| (vals[0], vals[1]))
        {
            let est_curr = self.est_times[disp_node_curr.est_idx.idx()];
            let est_next = self.est_times[disp_node_next.est_idx.idx()];

            if est_curr.idx_next == disp_node_next.est_idx
                && (est_next.idx_prev == disp_node_curr.est_idx
                    || est_next.idx_prev_alt == disp_node_curr.est_idx)
            {
                if !utils::almost_eq_uom(
                    &est_curr.dist_to_next,
                    &(disp_node_next.offset - disp_node_curr.offset),
                    None,
                ) {
                    errors.push(anyhow!(
                        "Dispatch path for train {} had an incorrect offset 
                        change of {:?} compared to {:?} at dispatch node {:?}!",
                        self.train_idx.idx(),
                        disp_node_next.offset - disp_node_curr.offset,
                        est_curr.dist_to_next,
                        disp_node_curr
                    ));
                }
            } else if est_curr.idx_next_alt == disp_node_next.est_idx
                && est_next.idx_prev == disp_node_curr.est_idx
            {
                if !utils::almost_eq_uom(&disp_node_curr.offset, &disp_node_next.offset, None) {
                    errors.push(anyhow!(
                        "Dispatch path for train {} had an incorrect offset change
                        at fake dispatch node {:?}!",
                        self.train_idx.idx(),
                        disp_node_curr
                    ));
                }
            } else {
                errors.push(anyhow!(
                    "Bad estimated time linking on dispatch path for train {}!",
                    self.train_idx.idx()
                ));
            }
            if !self.est_time_statuses[disp_node_curr.est_idx.idx()].is_on_path {
                errors.push(anyhow!(
                    "Est time status {:?} was not marked as being on the path
                    even though dispatch node {:?} is on the path!",
                    self.est_time_statuses[disp_node_curr.est_idx.idx()],
                    disp_node_curr
                ));
            }
        }

        //TODO: fix the occupancy problem.
        // Check for occupancy conflicts
        for disp_node_curr in &self.disp_path[self.disp_node_idx_free.idx()..] {
            if disp_node_curr.link_event.est_type == EstType::Arrive
                && links_blocked[disp_node_curr.link_event.link_idx.idx()].is_some()
            {
                errors.push(anyhow!(
                    "Occupancy conflict at link {} between train {} and train {} at dispatch node {:?}!",
                    disp_node_curr.link_event.link_idx.idx(),
                    self.train_idx.idx(),
                    links_blocked[disp_node_curr.link_event.link_idx.idx()].idx(),
                    disp_node_curr
                ));
            }
        }

        errors.make_err()
    }
}
