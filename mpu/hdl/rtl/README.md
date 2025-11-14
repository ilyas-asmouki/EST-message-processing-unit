# RTL Sources

`rtl/` contains only synthesizable code. Subdirectories reflect the flow of the transmit chain:

- `tx_chain/`: RS encoder, interleaver, scrambler, convolutional encoder, differential encoder, modulation prep, etc.
- `common/`: shared infrastructure (clock-domain crossing, FIFOs, register blocks, coeff ROMs).
- `top/`: device-specific wrappers that connect the chain to IO (DAC, SERDES, system bus).

Each block will get its own README and lint/constraint collateral as we implement it, but for now this folder is just a placeholder so we can start dropping modules in pipeline order when ready.
