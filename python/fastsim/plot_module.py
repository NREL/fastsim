import matplotlib.pyplot as plt
import numpy as np
from fuzzywuzzy import fuzz, process



available_signals = {
   'Fuel Converter Output Power Achieved': 'fc_kw_out_ach',
   'ESS Output Power Achieved': 'ess_kw_out_ach',
   'Fuel Converter Input Power Achieved': 'fc_kw_in_ach',
   'Acceleration Buffer SOC': 'accel_buff_soc',
   'Acceleration Power': 'accel_kw',
   'Ascent Energy (KJ)': 'ascent_kj',
   'Ascent Power': 'ascent_kw',
   'Auxiliary Power Input': 'aux_in_kw',
   'Auxiliary Energy (KJ)': 'aux_kj',
   'Battery Energy Consumption per Mile': 'battery_kwh_per_mi',
   'Brake Energy (KJ)': 'brake_kj',
   'Current ESS Max Output Power': 'cur_ess_max_kw_out',
   'Current Max Available Electric Power': 'cur_max_avail_elec_kw',
   'Current Max Electric Power': 'cur_max_elec_kw',
   'Current Max ESS Charge Power': 'cur_max_ess_chg_kw',
   'Current Max Fuel Converter Output Power': 'cur_max_fc_kw_out',
   'Current Max Fuel System Output Power': 'cur_max_fs_kw_out',
   'Current Max Motor Controller Input Power': 'cur_max_mc_elec_kw_in',
   'Current Max Motor Controller Output Power': 'cur_max_mc_kw_out',
   'Current Max Mechanical Motor Controller Input Power': 'cur_max_mech_mc_kw_in',
   'Current Max Roadway Charge Power': 'cur_max_roadway_chg_kw',
   'Current Max Traction Power': 'cur_max_trac_kw',
   'Current Max Transmission Output Power': 'cur_max_trans_kw_out',
   'Current SOC Target': 'cur_soc_target',
   'Cycle Friction Brake Power': 'cyc_fric_brake_kw',
   'Cycle Met': 'cyc_met',
   'Cycle Regen Brake Power': 'cyc_regen_brake_kw',
   'Cycle Tire Inertia Power': 'cyc_tire_inertia_kw',
   'Cycle Traction Power Required': 'cyc_trac_kw_req',
   'Cycle Transmission Output Power Required': 'cyc_trans_kw_out_req',
   'Cycle Wheel Power Required': 'cyc_whl_kw_req',
   'Cycle Wheel Rad Per Sec': 'cyc_whl_rad_per_sec',
   'Desired ESS Output Power for AE': 'desired_ess_kw_out_for_ae',
   'Distance (m)': 'dist_m',
   'Distance (mi)': 'dist_mi',
   'DOD Cycles': 'dod_cycs',
   'Drag Energy (KJ)': 'drag_kj',
   'Drag Power': 'drag_kw',
   'Electric Power Required for AE': 'elec_kw_req_4ae',
   'Electric Energy Consumption per Mile': 'electric_kwh_per_mi',
   'Energy Audit Error': 'energy_audit_error',
   'ER AE Output Power': 'er_ae_kw_out',
   'ER Power If Fuel Cell Required': 'er_kw_if_fc_req',
   'ESS to Fuel Energy (KWH)': 'ess2fuel_kwh',
   'ESS Acceleration Buffer Charge Power': 'ess_accel_buff_chg_kw',
   'ESS Acceleration Regen Discharge Power': 'ess_accel_regen_dischg_kw',
   'ESS AE Output Power': 'ess_ae_kw_out',
   'ESS Capacity Limit Charge Power': 'ess_cap_lim_chg_kw',
   'ESS Capacity Limit Discharge Power': 'ess_cap_lim_dischg_kw',
   'ESS Current Energy (KWH)': 'ess_cur_kwh',
   'ESS Desired Power for FC Efficiency': 'ess_desired_kw_4fc_eff',
   'ESS Discharge Energy (KJ)': 'ess_dischg_kj',
   'ESS Efficiency Energy (KJ)': 'ess_eff_kj',
   'ESS Power If Fuel Cell Required': 'ess_kw_if_fc_req',
   'ESS Lim Motor Controller Regen Percent Power': 'ess_lim_mc_regen_perc_kw',
   'ESS Loss Power': 'ess_loss_kw',
   'ESS Percentage Dead': 'ess_perc_dead',
   'ESS Regen Buffer Discharge Power': 'ess_regen_buff_dischg_kw',
   'Fuel Converter Energy (KJ)': 'fc_kj',
   'Fuel Converter Power Gap from Efficiency': 'fc_kw_gap_fr_eff',
   'Fuel Converter Output Power Achieved Percentage': 'fc_kw_out_ach_pct',
   'Fuel Converter Time On': 'fc_time_on',
   'Fuel Converter Transmission Limit Power': 'fc_trans_lim_kw',
   'Fuel System Cumulative MJ Output Achieved': 'fs_cumu_mj_out_ach',
   'Fuel System Output Power Achieved': 'fs_kw_out_ach',
   'Fuel System Output Energy (KWH) Achieved': 'fs_kwh_out_ach',
   'Fuel Energy (KJ)': 'fuel_kj',
   'Gap to Lead Vehicle (m)': 'gap_to_lead_vehicle_m',
   'IDM Target Speed (m/s)': 'idm_target_speed_m_per_s',
   'Kinetic Energy (KJ)': 'ke_kj',
   'Max ESS Acceleration Buffer Discharge Power': 'max_ess_accell_buff_dischg_kw',
   'Max ESS Regen Buffer Charge Power': 'max_ess_regen_buff_chg_kw',
   'Max Traction Speed (m/s)': 'max_trac_mps',
   'Motor Controller Input Power for Max FC Efficiency': 'mc_elec_in_kw_for_max_fc_eff',
   'Motor Controller Input Power Limit': 'mc_elec_in_lim_kw',
   'Motor Controller Input Power Achieved': 'mc_elec_kw_in_ach',
   'Motor Controller Input Power If Fuel Cell Required': 'mc_elec_kw_in_if_fc_req',
   'Motor Controller Energy (KJ)': 'mc_kj',
   'Motor Controller Power If Fuel Cell Required': 'mc_kw_if_fc_req',
   'Motor Controller Mechanical Power Output Achieved': 'mc_mech_kw_out_ach',
   'Motor Controller Transition Limit Power': 'mc_transi_lim_kw',
   'Min ESS Power to Help FC': 'min_ess_kw_2help_fc',
   'Min Motor Controller Power to Help FC': 'min_mc_kw_2help_fc',
   'MPGGE': 'mpgge',
   'Achieved Speed (MPH)': 'mph_ach',
   'Achieved Speed (m/s)': 'mps_ach',
   'Net Energy (KJ)': 'net_kj',
   'Regen Buffer SOC': 'regen_buff_soc',
   'Regen Control Limit KW Percentage': 'regen_contrl_lim_kw_perc',
   'Roadway Charge Energy (KJ)': 'roadway_chg_kj',
   'Roadway Charge Output Power Achieved': 'roadway_chg_kw_out_ach',
   'Rolling Resistance Energy (KJ)': 'rr_kj',
   'Rolling Resistance Power': 'rr_kw',
   'State of Charge (SOC)': 'soc',
   'Trace Miss': 'trace_miss',
   'Trace Miss Distance Fraction': 'trace_miss_dist_frac',
   'Trace Miss Iterations': 'trace_miss_iters',
   'Trace Miss Speed (m/s)': 'trace_miss_speed_mps',
   'Trace Miss Time Fraction': 'trace_miss_time_frac',
   'Transmission Energy (KJ)': 'trans_kj',
   'Transmission Input Power Achieved': 'trans_kw_in_ach',
   'Transmission Output Power Achieved': 'trans_kw_out_ach'
}




# Function for fuzzy matching signals
def fuzzy_match(signal_name, available_signals, feeling_lucky=False):
   matches = process.extractOne(signal_name, available_signals.keys(), scorer=fuzz.token_sort_ratio)
   best_match, score = matches
   if score >= 70:
       if feeling_lucky:
           return best_match
       else:
           print(f"Did you mean '{best_match}'? (y/n)")
           choice = input().strip().lower()
           if choice == 'y':
               return best_match
   return None




def plot(self, signal, fuzzy_search=True, feeling_lucky=False, speed_trace=True, difference=None, type='temporal'):
   # Ensure signal is a list
   if isinstance(signal, str):
       signal = [signal]


   # Handle fuzzy search
   if fuzzy_search:
       matched_signals = []
       for sig in signal:
           matched_signal = fuzzy_match(sig, available_signals, feeling_lucky)
           if matched_signal:
               matched_signals.append(matched_signal)
           else:
               print(f"No close match found for '{sig}'. Please enter a valid signal name.")
               return None
   else:
       matched_signals = signal


   # Prepare data for plotting
   plot_data = []
   for sig in matched_signals:
       if sig in available_signals:
           signal_key = available_signals[sig]
           # Assuming the data is available in self after simulation
           plot_data.append((getattr(self, signal_key), sig))
       else:
           print(f"Invalid signal name: '{sig}'")
           return None


   # Plotting
   if type == 'temporal':
       fig, axs = plt.subplots(len(plot_data) + (1 if speed_trace else 0), 1, figsize=(12, 8), sharex=True)
      
       for idx, (data, label) in enumerate(plot_data):
           axs[idx].plot(self.cyc.time_s, data, label=label)
           axs[idx].set_ylabel(label)
      
       if speed_trace:
           axs[-1].plot(self.cyc.time_s, self.mph_ach, label='Achieved Speed (MPH)', color='black')
           axs[-1].set_ylabel('Achieved Speed (MPH)')
      
       fig.suptitle('Temporal Performance Comparison')
       plt.xlabel('Time (s)')
       plt.legend()
       plt.tight_layout()
       plt.show()


   elif type == 'spatial':
       fig, ax = plt.subplots(figsize=(12, 8))
      
       # Assuming 'dist_m' is the spatial distance data
       for idx, (data, label) in enumerate(plot_data):
           ax.plot(self.dist_m, data, label=label)
      
       ax.set_xlabel('Distance (m)')
       ax.set_ylabel('Signal Value')
       ax.set_title('Spatial Performance Comparison')
       ax.legend()
       plt.tight_layout()
       plt.show()
  
   else:
       print("Invalid type selected. Please choose 'temporal' or 'spatial'.")
       return None

