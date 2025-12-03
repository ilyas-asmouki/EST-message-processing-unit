# ----------------------------------------------------------------------------
# AD9122 DAC interface (FMCOMMS1 on zedboard)
# ----------------------------------------------------------------------------

# data clock input (DCI) - driven by FPGA to DAC
set_property -dict {PACKAGE_PIN E19 IOSTANDARD LVDS_25} [get_ports dac_dci_p]
set_property -dict {PACKAGE_PIN E20 IOSTANDARD LVDS_25} [get_ports dac_dci_n]

# frame sync
set_property -dict {PACKAGE_PIN N17 IOSTANDARD LVDS_25} [get_ports dac_frame_p]
set_property -dict {PACKAGE_PIN N18 IOSTANDARD LVDS_25} [get_ports dac_frame_ n]

# data lines
set_property -dict {PACKAGE_PIN A21 IOSTANDARD LVDS_25} [get_ports {dac_d_p[0]}]
set_property -dict {PACKAGE_PIN A22 IOSTANDARD LVDS_25} [get_ports {dac_d_n[0]}]
set_property -dict {PACKAGE_PIN B21 IOSTANDARD LVDS_25} [get_ports {dac_d_p[1]}]
set_property -dict {PACKAGE_PIN B22 IOSTANDARD LVDS_25} [get_ports {dac_d_n[1]}]
set_property -dict {PACKAGE_PIN C15 IOSTANDARD LVDS_25} [get_ports {dac_d_p[2]}]
set_property -dict {PACKAGE_PIN B15 IOSTANDARD LVDS_25} [get_ports {dac_d_n[2]}]
set_property -dict {PACKAGE_PIN A16 IOSTANDARD LVDS_25} [get_ports {dac_d_p[3]}]
set_property -dict {PACKAGE_PIN A17 IOSTANDARD LVDS_25} [get_ports {dac_d_n[3]}]
set_property -dict {PACKAGE_PIN B16 IOSTANDARD LVDS_25} [get_ports {dac_d_p[4]}]
set_property -dict {PACKAGE_PIN B17 IOSTANDARD LVDS_25} [get_ports {dac_d_n[4]}]
set_property -dict {PACKAGE_PIN C17 IOSTANDARD LVDS_25} [get_ports {dac_d_p[5]}]
set_property -dict {PACKAGE_PIN C18 IOSTANDARD LVDS_25} [get_ports {dac_d_n[5]}]
set_property -dict {PACKAGE_PIN A18 IOSTANDARD LVDS_25} [get_ports {dac_d_p[6]}]
set_property -dict {PACKAGE_PIN A19 IOSTANDARD LVDS_25} [get_ports {dac_d_n[6]}]
set_property -dict {PACKAGE_PIN D22 IOSTANDARD LVDS_25} [get_ports {dac_d_p[7]}]
set_property -dict {PACKAGE_PIN C22 IOSTANDARD LVDS_25} [get_ports {dac_d_n[7]}]
set_property -dict {PACKAGE_PIN G19 IOSTANDARD LVDS_25} [get_ports {dac_d_p[8]}]
set_property -dict {PACKAGE_PIN F19 IOSTANDARD LVDS_25} [get_ports {dac_d_n[8]}]
set_property -dict {PACKAGE_PIN E21 IOSTANDARD LVDS_25} [get_ports {dac_d_p[9]}]
set_property -dict {PACKAGE_PIN D21 IOSTANDARD LVDS_25} [get_ports {dac_d_n[9]}]
set_property -dict {PACKAGE_PIN F18 IOSTANDARD LVDS_25} [get_ports {dac_d_p[10]}]
set_property -dict {PACKAGE_PIN E18 IOSTANDARD LVDS_25} [get_ports {dac_d_n[10]}]
set_property -dict {PACKAGE_PIN E15 IOSTANDARD LVDS_25} [get_ports {dac_d_p[11]}]
set_property -dict {PACKAGE_PIN D15 IOSTANDARD LVDS_25} [get_ports {dac_d_n[11]}]
set_property -dict {PACKAGE_PIN G15 IOSTANDARD LVDS_25} [get_ports {dac_d_p[12]}]
set_property -dict {PACKAGE_PIN G16 IOSTANDARD LVDS_25} [get_ports {dac_d_n[12]}]
set_property -dict {PACKAGE_PIN G20 IOSTANDARD LVDS_25} [get_ports {dac_d_p[13]}]
set_property -dict {PACKAGE_PIN G21 IOSTANDARD LVDS_25} [get_ports {dac_d_n[13]}]
set_property -dict {PACKAGE_PIN J16 IOSTANDARD LVDS_25} [get_ports {dac_d_p[14]}]
set_property -dict {PACKAGE_PIN J17 IOSTANDARD LVDS_25} [get_ports {dac_d_n[14]}]
set_property -dict {PACKAGE_PIN J20 IOSTANDARD LVDS_25} [get_ports {dac_d_p[15]}]
set_property -dict {PACKAGE_PIN K21 IOSTANDARD LVDS_25} [get_ports {dac_d_n[15]}]

# optional: referfence clock input from FMC (if used)
# set_property -dict {PACKAGE_PIN L18 IOSTANDARD LVDS_25 DIFF_TERM TRUE} [get_ports dac_clk_in_p]
# set_property -dict {PACKAGE_PIN L19 IOSTANDARD LVDS_25 DIFF_TERM TRUE} [get_ports dac_clk_in_n]
