import numpy as np
import matplotlib.pyplot as plt
from mpu.model.rrc import _rrc_impulse_response

beta, sps, span = 0.35, 8, 8
h = _rrc_impulse_response(beta, sps, span, normalize="unit-energy")

n = np.arange(-(len(h)//2), len(h)//2 + 1)   # centered sample index
t_sym = n / sps                               # time in *symbol* durations

plt.figure()
plt.plot(t_sym, h)
plt.title(f"RRC impulse (β={beta}, sps={sps}, span={span})")
plt.xlabel("time [symbols]")
plt.ylabel("amplitude")
plt.grid(True)
plt.show()
