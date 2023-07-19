use super::super::disp_imports::*;
use super::TrainDisp;

impl TrainDisp {
    /// *Note*:
    ///
    /// This can be passed into `rustc` via --cfg flag, e.g.:
    ///
    /// `rustc --cfg debug_advance_rewind train_disp.rs`,
    ///
    /// To use via cargo, create a `.cargo/config.toml` with the
    /// following contents:
    ///
    /// \[build\]
    ///
    /// rustflags = "--cfg debug_advance_rewind"
    pub fn advance(
        &mut self,
        link_disp_auths: &mut [Vec<DispAuth>],
        links_blocked: &mut [TrainIdx],
        links: &[Link],
    ) -> bool {
        let time_overlap_change = 30.0 * uc::S;

        assert!(self.disp_node_idx_free.idx() < self.disp_path.len());
        assert!(
            self.disp_path[self.disp_node_idx_free.idx()]
                .time_pass
                .is_infinite(),
            "Train {} has a timed free node!",
            self.train_idx.idx()
        );

        // Update is_blocked
        self.is_blocked = if self.disp_node_idx_front.is_some() {
            let disp_node_front = &self.disp_path[self.disp_node_idx_front.idx()];
            let link_idx_front = disp_node_front.link_event.link_idx;
            let link_disp_front = &link_disp_auths[link_idx_front.idx()];
            let disp_auth_idx_curr = disp_node_front.disp_auth_idx_entry;

            debug_assert!(link_idx_front.is_real());
            debug_assert!(disp_auth_idx_curr.is_some());
            debug_assert!(self.train_idx == link_disp_front[disp_auth_idx_curr.idx()].train_idx);
            debug_assert!(
                (self.offset_free - disp_node_front.offset)
                    <= link_disp_front[disp_auth_idx_curr.idx() - 1].offset_back
            );

            !link_disp_front[disp_auth_idx_curr.idx() - 1]
                .offset_back
                .is_infinite()
        } else {
            false
        };

        // Update time_update_next
        self.time_update_next = if self.disp_node_idx_free.is_some()
            && self.disp_node_idx_free.idx() < self.disp_path.len()
        {
            let time_update = self.disp_path[self.disp_node_idx_free.idx() - 1].time_pass;

            let est_time_prev = &self.est_times[self.disp_path[self.disp_node_idx_free.idx() - 1]
                .est_idx
                .idx()];

            if est_time_prev.idx_next == self.disp_path[self.disp_node_idx_free.idx()].est_idx {
                time_update + est_time_prev.time_to_next
            } else {
                time_update
            }
        } else {
            self.time_update
        };

        let disp_node_idx_save = self.disp_node_idx_free;
        let offset_save = self.offset_free;

        // Advance forward one node at a time
        loop {
            let disp_node_curr = &self.disp_path[self.disp_node_idx_free.idx()];
            let est_time_curr = &self.est_times[disp_node_curr.est_idx.idx()];

            // If this is the last real dispatch node before a blockage, fix the free node and break
            if self.is_blocked && self.offset_free <= disp_node_curr.offset {
                while self.disp_node_idx_free.idx() < self.disp_path.len() {
                    let disp_node_free = &self.disp_path[self.disp_node_idx_free.idx()];
                    if self.offset_free < disp_node_free.offset
                        || disp_node_free.link_event.est_type != EstType::Fake
                    {
                        break;
                    }
                    self.disp_node_idx_free =
                        (self.disp_node_idx_free.idx() + 1).try_from_idx().unwrap();
                }
                break;
            }

            let link_idx_curr = disp_node_curr.link_event.link_idx;
            let link_curr = &links[link_idx_curr.idx()];

            // Handle an arrive event
            if disp_node_curr.link_event.est_type == EstType::Arrive {
                let link_idxs_lockout = &link_curr.link_idxs_lockout;

                // Note:  can be passed into `rustc` via --cfg flag, e.g.:
                // `rustc --cfg debug_advance_rewind train_disp.rs`,
                // To use via cargo, create a `.cargo/config.toml` with the
                // following contents:
                // [build]
                // rustflags = "--cfg debug_advance_rewind"
                // TODO:  Might be simpler to make this a feature, if it's still necessary
                // Verify that free path is in fact free
                #[cfg(debug_advance_rewind)]
                {
                    for link_idx_lockout in link_idxs_lockout {
                        assert!(link_disp_auths[link_idx_lockout.idx()]
                            .last()
                            .unwrap()
                            .offset_back
                            .is_infinite());
                    }
                    assert!(link_disp_auths[link_curr.idx_flip.idx()]
                        .last()
                        .unwrap()
                        .offset_back
                        .is_infinite());
                }

                let disp_auth_prev = link_disp_auths[link_idx_curr.idx()].last().unwrap();

                // If the link cannot be exited, limit the new auth offset
                if disp_auth_prev.offset_back.is_finite() {
                    self.offset_free = disp_node_curr.offset + disp_auth_prev.offset_back;
                    self.is_blocked = true;
                    // If the link cannot be entered, break
                    if disp_auth_prev.offset_back == si::Length::ZERO {
                        break;
                    }
                }

                // If the train is arriving at a turnout or diamond, pause
                if self.offset_free < disp_node_curr.offset
                    && (!link_idxs_lockout.is_empty()
                        || links[link_curr.idx_next.idx()].idx_prev_alt.is_real())
                {
                    self.offset_free = disp_node_curr.offset;
                    break;
                }

                // Update time next based on current disp_auths
                let time_startup = est_time_curr.velocity / self.acc_startup;
                let flip_clear_exit = link_disp_auths[link_curr.idx_flip.idx()]
                    .last()
                    .unwrap()
                    .clear_exit;
                // If the train is going the same direction as the previous train
                let time_update_max = if disp_auth_prev.clear_exit >= flip_clear_exit {
                    disp_auth_prev.clear_entry + self.time_spacing
                }
                // Otherwise, the train was going the opposite direction
                else {
                    flip_clear_exit + time_startup
                };
                self.time_update_next = self.time_update_next.max(time_update_max);

                // Update time next based on link idxs lockout
                if link_idx_curr.is_real() {
                    self.link_idxs_blocking.push(link_curr.idx_flip);

                    for link_idx_lockout in link_idxs_lockout {
                        self.time_update_next = self.time_update_next.max(
                            link_disp_auths[link_idx_lockout.idx()]
                                .last()
                                .unwrap()
                                .clear_exit
                                + time_overlap_change
                                + time_startup,
                        );
                    }
                    self.link_idxs_blocking.extend(link_idxs_lockout);
                }

                // Move front of train out of previously entered link_idx
                if self.disp_node_idx_front.is_some() {
                    let disp_node_exit = &self.disp_path[self.disp_node_idx_front.idx()];
                    let link_idx_exit = disp_node_exit.link_event.link_idx;
                    let disp_auth_idx_exit = disp_node_exit.disp_auth_idx_entry;

                    debug_assert!(link_idx_exit.is_real());
                    debug_assert!(disp_auth_idx_exit.is_some());

                    let disp_auths_exit = &mut link_disp_auths[link_idx_exit.idx()];
                    let disp_auth_same_dir_exit = &disp_auths_exit[disp_auth_idx_exit.idx() - 1];
                    debug_assert!(disp_auth_same_dir_exit.offset_back.is_infinite());

                    self.time_update_next = self
                        .time_update_next
                        .max(disp_auth_same_dir_exit.clear_exit + self.time_spacing);

                    let disp_auth_exit = &mut disp_auths_exit[disp_auth_idx_exit.idx()];
                    debug_assert!(disp_auth_exit.train_idx == self.train_idx);

                    disp_auth_exit.offset_front = f64::INFINITY * uc::M;
                    disp_auth_exit.arrive_exit = self.time_update_next;
                }

                // Enter the next link_idx
                if link_idx_curr.is_real() {
                    let disp_auths_curr = &mut link_disp_auths[link_idx_curr.idx()];
                    self.disp_path[self.disp_node_idx_free.idx()].disp_auth_idx_entry =
                        disp_auths_curr.len().try_from_idx().unwrap();
                    disp_auths_curr.push(DispAuth {
                        arrive_entry: self.time_update_next,
                        train_idx: self.train_idx,
                        ..Default::default()
                    });
                }
                self.disp_node_idx_front = self.disp_node_idx_free;
            }
            // Handle a clear event
            else if disp_node_curr.link_event.est_type == EstType::Clear {
                // Move back of train out of previously cleared link_idx
                if self.disp_node_idx_back.is_some() {
                    let disp_node_exit = &self.disp_path[self.disp_node_idx_back.idx()];
                    let link_idx_exit = disp_node_exit.link_event.link_idx;
                    let disp_auth_idx_exit = disp_node_exit.disp_auth_idx_entry;

                    debug_assert!(link_idx_exit.is_real());
                    debug_assert!(disp_auth_idx_exit.is_some());

                    let disp_auths_exit = &mut link_disp_auths[link_idx_exit.idx()];
                    let disp_auth_same_dir_exit = &disp_auths_exit[disp_auth_idx_exit.idx() - 1];
                    debug_assert!(disp_auth_same_dir_exit.offset_back.is_infinite());

                    // Clear link and adjust links_blocked
                    let disp_auth_exit = &mut disp_auths_exit[disp_auth_idx_exit.idx()];
                    debug_assert!(disp_auth_exit.train_idx == self.train_idx);

                    disp_auth_exit.offset_back = f64::INFINITY * uc::M;
                    disp_auth_exit.clear_exit = self.time_update_next;
                    update_links_blocked(
                        links_blocked,
                        links,
                        link_idx_exit,
                        disp_auths_exit.last().unwrap().train_idx_curr(),
                    );

                    // Remove link_idxs_blocking corresponding to link_exit
                    self.link_idxs_blocking
                        .drain(..1 + links[link_idx_exit.idx()].link_idxs_lockout.len());
                }

                // Update disp_auth_idx_entry, clear_entry, and disp_node_idx_back
                if link_idx_curr.is_real() {
                    let disp_auth_idx =
                        &mut self.disp_path[self.disp_node_idx_free.idx()].disp_auth_idx_entry;
                    let disp_auths_curr = &mut link_disp_auths[link_idx_curr.idx()];
                    *disp_auth_idx = (disp_auths_curr.len() - 1).try_from_idx().unwrap();

                    let disp_auth_curr = &mut disp_auths_curr[disp_auth_idx.idx()];
                    debug_assert!(disp_auth_curr.train_idx == self.train_idx);

                    disp_auth_curr.clear_entry = self.time_update_next;
                    self.disp_node_idx_back = self.disp_node_idx_free;
                }
            }

            self.disp_path[self.disp_node_idx_free.idx()].time_pass = self.time_update_next;
            self.disp_node_idx_free = (self.disp_node_idx_free.idx() + 1).try_from_idx().unwrap();

            if self.disp_node_idx_free.idx() == self.disp_path.len() {
                // TODO:  Don't use unwrap here
                self.offset_free = self.disp_path.last().unwrap().offset;
                break;
            }

            if est_time_curr.idx_next == self.disp_path[self.disp_node_idx_free.idx()].est_idx {
                self.time_update_next += est_time_curr.time_to_next;
            }
        }

        // If the train did not move, return false and do not update occupancy
        if self.offset_free == offset_save && self.disp_node_idx_free == disp_node_idx_save {
            false
        } else {
            self.update_occupancy(link_disp_auths, links_blocked, links);
            true
        }
    }

    pub fn rewind(
        &mut self,
        link_disp_auths: &mut [Vec<DispAuth>],
        links_blocked: &mut [TrainIdx],
        links: &[Link],
    ) {
        assert!(
            self.disp_node_idx_free.idx() < self.disp_path.len(),
            "Train {} cannot rewind after exiting!",
            self.train_idx.idx()
        );
        assert!(self.disp_node_idx_fixed <= self.disp_node_idx_free);
        assert!(
            self.disp_path[self.disp_node_idx_fixed.idx()].offset >= self.offset_fixed,
            "Train {} cannot be rewound with an invalid new offset and dispatch node combo!",
            self.train_idx.idx()
        );

        self.offset_free = self.offset_fixed;

        while self.disp_node_idx_fixed < self.disp_node_idx_free {
            self.disp_node_idx_free = (self.disp_node_idx_free.idx() - 1).try_from_idx().unwrap();

            self.disp_path[self.disp_node_idx_free.idx()].time_pass = f64::INFINITY * uc::S;

            let (link_idx_curr, est_type_curr) = {
                let link_event_curr = self.disp_path[self.disp_node_idx_free.idx()].link_event;
                (link_event_curr.link_idx, link_event_curr.est_type)
            };

            // Handle arrive event
            if est_type_curr == EstType::Arrive {
                assert!(link_idx_curr.is_real());
                debug_assert!(self.disp_node_idx_front == self.disp_node_idx_free);

                // Adjust disp_node_idx_front
                loop {
                    self.disp_node_idx_front =
                        (self.disp_node_idx_front.idx() - 1).try_from_idx().unwrap();
                    if self.disp_path[self.disp_node_idx_front.idx()]
                        .link_event
                        .est_type
                        == EstType::Arrive
                        || self.disp_node_idx_front.is_none()
                    {
                        break;
                    }
                }

                // Remove the entered dispatch authority
                let disp_auths_exit = &mut link_disp_auths[link_idx_curr.idx()];
                let disp_auth_idx_entry =
                    &mut self.disp_path[self.disp_node_idx_free.idx()].disp_auth_idx_entry;
                debug_assert!(disp_auth_idx_entry.idx() == disp_auths_exit.len() - 1);
                disp_auths_exit.pop();
                *disp_auth_idx_entry = None;

                // Update the link train idxs
                update_links_blocked(
                    links_blocked,
                    links,
                    link_idx_curr,
                    disp_auths_exit.last().unwrap().train_idx_curr(),
                );

                // Remove the corresponding entered link_idxs_blocking
                let new_len = self.link_idxs_blocking.len()
                    - 1
                    - links[link_idx_curr.idx()].link_idxs_lockout.len();
                self.link_idxs_blocking.truncate(new_len);

                // If the front of the train is still occupying a link
                if self.disp_node_idx_front.is_some() {
                    let disp_node_front = &self.disp_path[self.disp_node_idx_front.idx()];
                    let link_front = &links[disp_node_front.link_event.link_idx.idx()];

                    // Validate that the link_idxs_blocking for the front node are correct
                    debug_assert!(
                        link_front.link_idxs_lockout
                            == self.link_idxs_blocking[self.link_idxs_blocking.len()
                                - link_front.link_idxs_lockout.len()..]
                    );
                    debug_assert!(
                        link_front.idx_flip
                            == self.link_idxs_blocking[self.link_idxs_blocking.len()
                                - 1
                                - link_front.link_idxs_lockout.len()]
                    );

                    // Reset the arrive_exit time for the front node
                    link_disp_auths[disp_node_front.link_event.link_idx.idx()]
                        [disp_node_front.disp_auth_idx_entry.idx()]
                    .arrive_exit = f64::INFINITY * uc::S;
                } else {
                    debug_assert!(self.link_idxs_blocking.is_empty());
                }
            }
            // Handle clear event
            else if est_type_curr == EstType::Clear {
                assert!(link_idx_curr.is_real());
                debug_assert!(self.disp_node_idx_back == self.disp_node_idx_free);

                // Adjust disp_node_idx_back
                loop {
                    self.disp_node_idx_back =
                        (self.disp_node_idx_back.idx() - 1).try_from_idx().unwrap();

                    if self.disp_path[self.disp_node_idx_back.idx()]
                        .link_event
                        .est_type
                        == EstType::Clear
                        || self.disp_node_idx_back.is_none()
                    {
                        break;
                    }
                }

                // Re-insert the previously cleared link_idxs_blocking
                if self.disp_node_idx_back.is_some() {
                    let disp_node_back = &self.disp_path[self.disp_node_idx_back.idx()];
                    let link_entry = &links[disp_node_back.link_event.link_idx.idx()];

                    // Insert correct number of link_idxs into link_idxs_blocking
                    self.link_idxs_blocking.splice(
                        0..0,
                        vec![link_entry.idx_flip; 1 + link_entry.link_idxs_lockout.len()],
                    );

                    // Overwrite appropriate link_idxs_blocking with link_idxs_lockout
                    self.link_idxs_blocking[1..1 + link_entry.link_idxs_lockout.len()]
                        .copy_from_slice(&link_entry.link_idxs_lockout);

                    // Reset clear exit time
                    link_disp_auths[disp_node_back.link_event.link_idx.idx()]
                        [disp_node_back.disp_auth_idx_entry.idx()]
                    .clear_exit = f64::INFINITY * uc::S;
                }

                // Reset back clear entry event
                let disp_auths_exit = &mut link_disp_auths[link_idx_curr.idx()];
                let disp_auth_idx_entry =
                    &mut self.disp_path[self.disp_node_idx_free.idx()].disp_auth_idx_entry;
                debug_assert!(disp_auth_idx_entry.idx() == disp_auths_exit.len() - 1);

                disp_auths_exit.last_mut().unwrap().offset_back = si::Length::ZERO;
                disp_auths_exit.last_mut().unwrap().clear_entry = f64::INFINITY * uc::S;
                *disp_auth_idx_entry = None;
            }
        }

        self.update_occupancy(link_disp_auths, links_blocked, links);
    }

    // TODO: this needs documentation!
    fn update_occupancy(
        &mut self,
        link_disp_auths: &mut [Vec<DispAuth>],
        links_blocked: &mut [TrainIdx],
        links: &[Link],
    ) {
        // If the train has not left, set its occupancy
        if self.disp_node_idx_free.idx() < self.disp_path.len() {
            // Update front occupancy
            if self.disp_node_idx_front.is_some() {
                let disp_node_front = &self.disp_path[self.disp_node_idx_front.idx()];
                let offset_front = self.offset_free - disp_node_front.offset;

                let disp_auths_front =
                    &mut link_disp_auths[disp_node_front.link_event.link_idx.idx()];
                disp_auths_front[disp_node_front.disp_auth_idx_entry.idx()].offset_front =
                    offset_front;

                let disp_auth_prev_train =
                    &disp_auths_front[disp_node_front.disp_auth_idx_entry.idx() - 1];
                debug_assert!(
                    offset_front <= disp_auth_prev_train.offset_back,
                    "The front of train {} was placed past the back of train {}!",
                    self.train_idx.idx(),
                    disp_auth_prev_train.train_idx.idx()
                );
            }

            // Update back occupancy
            if self.disp_node_idx_back.is_some() {
                let disp_node_back = &self.disp_path[self.disp_node_idx_back.idx()];
                let offset_back = self.offset_free - disp_node_back.offset;

                let disp_auths_back =
                    &mut link_disp_auths[disp_node_back.link_event.link_idx.idx()];
                disp_auths_back[disp_node_back.disp_auth_idx_entry.idx()].offset_back = offset_back;

                let disp_auth_idx_next = disp_node_back.disp_auth_idx_entry.idx() + 1;
                debug_assert!(
                    disp_auth_idx_next == disp_auths_back.len()
                        || disp_auths_back[disp_auth_idx_next].offset_front <= offset_back,
                    "The back of train {} was placed prior to the front of the next train {}!",
                    self.train_idx.idx(),
                    disp_auths_back[disp_auth_idx_next].train_idx.idx()
                );
            }

            // TODO: Check if necessary
            // Block appropriate links_blocked
            for link_idx in &self.link_idxs_blocking {
                links_blocked[link_idx.idx()] = self.train_idx;
            }
        }
        // If the train has left, remove any excess occupancy (WILL CAUSE ISSUES FOR EARLY EXIT)
        else if !self.link_idxs_blocking.is_empty() {
            assert!(self.disp_node_idx_back.is_some());
            let link_idx_back = self.disp_path[self.disp_node_idx_back.idx()]
                .link_event
                .link_idx;

            // Iterate backwards through the dispatch path
            for disp_node in self.disp_path.iter().rev() {
                // If the node is an arrive node, clear it
                if disp_node.link_event.est_type == EstType::Arrive {
                    assert!(disp_node.link_event.link_idx.is_real());

                    let disp_auths_exit = &mut link_disp_auths[disp_node.link_event.link_idx.idx()];
                    let disp_auth_exit = &mut disp_auths_exit[disp_node.disp_auth_idx_entry.idx()];
                    disp_auth_exit.offset_front = f64::INFINITY * uc::M;
                    disp_auth_exit.offset_back = f64::INFINITY * uc::M;
                    disp_auth_exit.arrive_exit =
                        disp_auth_exit.arrive_exit.min(self.time_update_next);
                    disp_auth_exit.arrive_entry =
                        disp_auth_exit.clear_entry.min(self.time_update_next);
                    disp_auth_exit.clear_exit = self.time_update_next;

                    update_links_blocked(
                        links_blocked,
                        links,
                        disp_node.link_event.link_idx,
                        disp_auths_exit.last().unwrap().train_idx_curr(),
                    );

                    // Exit if the disp node at the back of the train was cleared
                    if disp_node.link_event.link_idx == link_idx_back {
                        break;
                    }
                }
            }

            // Unblock all link_idxs
            self.link_idxs_blocking.clear();
        }
    }
}

/// Update links_blocked
fn update_links_blocked(
    links_blocked: &mut [TrainIdx],
    links: &[Link],
    link_idx: LinkIdx,
    train_idx_curr: TrainIdx,
) {
    let link_idx_flip = links[link_idx.idx()].idx_flip;
    // Set link_train_idx for current link and all lockouts
    links_blocked[link_idx_flip.idx()] = train_idx_curr;
    for link_idx_lockout in &links[link_idx.idx()].link_idxs_lockout {
        links_blocked[link_idx_lockout.idx()] = train_idx_curr;
    }

    // If this was a reset
    if train_idx_curr.is_none() {
        reset_link_train_idx(links_blocked, links, &link_idx_flip);
        for link_idx_lockout in &links[link_idx.idx()].link_idxs_lockout {
            reset_link_train_idx(links_blocked, links, link_idx_lockout);
        }
    }
}

/// Reset individual link_idx
fn reset_link_train_idx(links_blocked: &mut [TrainIdx], links: &[Link], link_idx: &LinkIdx) {
    // If other links could still be locking out the lockout
    if links[link_idx.idx()].link_idxs_lockout.len() > 1 {
        for link_idx_adjacent in &links[link_idx.idx()].link_idxs_lockout {
            // If the lockout is being locked out, update it and break
            if links_blocked[link_idx_adjacent.idx()].is_some() {
                links_blocked[link_idx.idx()] = links_blocked[link_idx_adjacent.idx()];
                break;
            }
        }
    }
}
