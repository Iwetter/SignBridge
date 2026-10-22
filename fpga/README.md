# Optional accelerator workstream — paused for CPU acceptance

No FPGA bitstream or physical acceleration result is delivered yet. The current host protocol and quantized Python model are integration groundwork only. RTL, simulation parity and board deployment remain incomplete.

The DE10-Standard is connected through USB-Blaster II and the onboard USB-to-UART socket. The onboard serial connection is to the HPS, not automatically to a user FPGA inference core. Do not point SignBridge binary packets at the normal Linux console. A verified HPS bridge or supported JTAG/System Console fabric interface is needed first.

Board pin assignments must come from the correct official DE10-Standard reference project/manual. No GPIO wiring assumptions are made. CPU remains the demo baseline and board work is deferred until the user has tested it.

`training/export_quantized.py` exports a 63→32→26 integer model, C-order weight memories and known numeric vectors. The protocol and exact arithmetic are defined in `docs/interface-contract.md`. Python quantized inference is a software oracle, not an FPGA benchmark.
