# EST-message-processing-unit

The plan going forward is to implement the following modules individually, test them in simulation, integrate them together and have the full FPGA close to ready by week 10.

---

## Weeks 1–2
**Documentation & Bring-up**
- Documentation, getting introduced with the project's building blocks  
- Verilog bring-up and project kick-off  

---

## Week 3
- Set up repo + testing environment (icarus)  
- Python model for:  
  - RS encoder (RS(255, 223))  
  - Conv encoder [171, 133]  
  - QPSK mapper (2 bits -> I/Q pair)  
- Implement HDL "skeletons" (empty modules with ports)  
- Test: feed "HELLO WORLD" through python models, get reference output (bits and I/Q)  
- *Optional*: need to test FPGA board basic functionality ASAP  

---

## Week 4: Front-end (RS + Interleaver)
- RS encoder in HDL  
- Interleaver in HDL (start with depth 1 = off, then 2, …)  
- Verify HDL outputs against python model bit for bit  
- **Milestone**: any input packet -> RS+interleaver HDL matches python  

---

## Week 5: Middle (Conv + Differential)
- Convolutional encoder in HDL  
- Differential encoder in HDL  
- Test both separately, then chain with RS+interleaver  
- **Milestone**: pipeline up to differential encoding fully tested in sim  

---

## Week 6: Back-end (QPSK + RRC)
- QPSK mapper in HDL  
- Implement Root-Raised Cosine filter (use python to generate coefficients)  
- Simulate chain: random packet -> RS -> Conv -> QPSK -> RRC  
- Compare FPGA sim output to python model  
- **Milestone**: full chain simulated, end-to-end  

---

## Week 7: Hardware bring-up
*(should ABSOLUTELY have FPGA by now)*  
- Synthesize RS -> Conv -> QPSK pipeline, load onto FPGA  
- Create a test setup that feeds packets from PC (via UART/Ethernet/USB) into FPGA  
- Verify FPGA outputs I/Q samples -> capture via logic analyzer or dev-board DAC  
- First “hello world” hardware demo: FPGA turns "HELLO" into I/Q symbols visible in a waveform viewer  

---

## Weeks 8–14: Integration
- Integration, solve bugs, test entire pipeline with OBC  
