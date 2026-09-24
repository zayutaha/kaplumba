# Technical Plan

We have hit a physical wall. The M4 Pro ALU/Gather throughput limits EXL3 decode to 300 Giga-weights/second. 
For 40 layers of 478.8M weights, 300 Gw/s equates to 63.8 ms. 
Adding `lm_head` (4.6 ms) makes the absolute theoretical minimum 68.4 ms (14.6 t/s).
20 t/s requires 50 ms.

We must pivot. I need the user to authorize lifting one of the standing constraints to proceed.
